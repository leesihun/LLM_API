"""
Chat completions endpoint (OpenAI-compatible with extensions)
/v1/chat/completions
"""
import json
import time
import uuid
from typing import Optional, List, Dict
from fastapi import APIRouter, Form, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from backend.models.schemas import (
    ChatMessage,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta
)
from backend.core.database import db, conversation_store
from backend.core.llm_backend import llm_backend
from backend.utils.file_handler import save_uploaded_files, read_file_content
from backend.utils.auth import get_optional_user
from backend.agents import ChatAgent, ReActAgent, PlanExecuteAgent, AutoAgent
from fastapi import Depends
import config

router = APIRouter(prefix="/v1", tags=["chat"])


def _get_agent(agent_type: str, model: str, temperature: float):
    """
    Get agent instance based on type

    Args:
        agent_type: Type of agent (chat, react, plan_execute, auto)
        model: Model name
        temperature: Temperature setting

    Returns:
        Agent instance
    """
    agent_type = agent_type.lower()

    if agent_type == "chat":
        return ChatAgent(model, temperature)
    elif agent_type == "react":
        return ReActAgent(model, temperature)
    elif agent_type == "plan_execute":
        return PlanExecuteAgent(model, temperature)
    elif agent_type == "auto":
        return AutoAgent(model, temperature)
    else:
        # Default to chat
        return ChatAgent(model, temperature)


def _prepare_messages_with_files(
    messages: List[ChatMessage],
    file_paths: List[str]
) -> List[Dict[str, str]]:
    """
    Prepare messages for LLM, adding file contents to context

    Args:
        messages: Chat messages
        file_paths: List of file paths to include

    Returns:
        List of message dictionaries with file contents added
    """
    # Convert Pydantic models to dicts
    message_dicts = [{"role": msg.role, "content": msg.content} for msg in messages]

    # If files are provided, add them to the last user message
    if file_paths:
        file_contents = []
        for file_path in file_paths:
            content = read_file_content(file_path)
            file_contents.append(f"\n\n--- File: {file_path} ---\n{content}")

        # Append to last user message
        if message_dicts and message_dicts[-1]["role"] == "user":
            message_dicts[-1]["content"] += "\n".join(file_contents)

    return message_dicts


@router.post("/chat/completions")
async def chat_completions(
    # Multipart form fields
    model: str = Form(...),
    messages: str = Form(...),  # JSON string
    stream: str = Form("false"),
    temperature: Optional[str] = Form(None),
    max_tokens: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    agent_type: str = Form("chat"),
    files: Optional[List[UploadFile]] = File(None),
    # Auth (optional)
    current_user: Optional[dict] = Depends(get_optional_user)
):
    """
    OpenAI-compatible chat completions endpoint with extensions

    Extensions:
    - Multipart/form-data support for file uploads
    - session_id for conversation continuity
    - agent_type for future agent selection
    - x_session_id in response for tracking

    Args:
        model: Model name
        messages: JSON string of message list
        stream: "true" or "false" for streaming
        temperature: Optional temperature override
        max_tokens: Optional max tokens override
        session_id: Optional session ID to continue conversation
        agent_type: Agent type (chat, auto, react, plan_execute)
        files: Optional file uploads
        current_user: Authenticated user (if any)
    """
    try:
        # Parse messages JSON
        messages_data = json.loads(messages)
        chat_messages = [ChatMessage(**msg) for msg in messages_data]

        # Convert stream string to boolean
        is_streaming = stream.lower() == "true"

        # Parse optional parameters
        temp = float(temperature) if temperature else config.DEFAULT_TEMPERATURE
        max_tok = int(max_tokens) if max_tokens else config.DEFAULT_MAX_TOKENS

        # Determine username (default to "guest" if not authenticated)
        username = current_user["username"] if current_user else "guest"

        # Handle session
        if session_id:
            # Continue existing session
            session = db.get_session(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")

            # Load conversation history
            history = conversation_store.load_conversation(session_id)
            if history is None:
                history = []

            # Add new messages to history
            for msg in chat_messages:
                history.append({"role": msg.role, "content": msg.content})

        else:
            # Create new session
            session_id = str(uuid.uuid4())
            db.create_session(session_id, username)
            history = [{"role": msg.role, "content": msg.content} for msg in chat_messages]

        # Handle file uploads
        file_paths = []
        if files and len(files) > 0:
            file_paths = save_uploaded_files(files, username, session_id)

        # Prepare messages with file contents (for conversation history)
        llm_messages = _prepare_messages_with_files(chat_messages, file_paths)

        # Get the appropriate agent
        agent = _get_agent(agent_type, model, temp)

        # Set session_id on agent for logging
        agent.session_id = session_id

        # Extract user input (last message) and conversation history (previous messages)
        # For continued sessions, use the full loaded history instead of just new messages
        if len(llm_messages) > 0:
            if session_id and len(history) > len(chat_messages):
                # Continued session - use full history from storage
                # Update the last message in history with file contents (if any)
                if file_paths:
                    # The last message in history is the new user message
                    # llm_messages[-1] has file contents appended
                    history[-1]["content"] = llm_messages[-1]["content"]

                # Extract user input (last message) and full conversation history
                user_input = history[-1]["content"]
                conversation_history = history[:-1]
            else:
                # New session - use llm_messages as before
                user_input = llm_messages[-1]["content"]
                conversation_history = llm_messages[:-1] if len(llm_messages) > 1 else []
        else:
            user_input = ""
            conversation_history = []

        # Generate response
        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created_timestamp = int(time.time())

        if is_streaming:
            # Streaming response
            # Note: Agents don't support streaming yet, fall back to direct LLM
            async def generate_stream():
                """Generator for SSE streaming"""
                try:
                    assistant_message = ""

                    # For streaming, prepare full message context
                    # Use the same logic as non-streaming to get full history
                    if session_id and len(history) > len(chat_messages):
                        # Continued session - use full history
                        stream_messages = history
                    else:
                        # New session - use llm_messages
                        stream_messages = llm_messages

                    # Stream tokens from LLM directly (bypass agents for streaming)
                    for token in llm_backend.chat_stream(stream_messages, model, temp, session_id=session_id, agent_type="stream"):
                        assistant_message += token

                        # Send SSE chunk
                        chunk = ChatCompletionChunk(
                            id=request_id,
                            created=created_timestamp,
                            model=model,
                            choices=[
                                ChatCompletionChunkChoice(
                                    delta=ChatCompletionChunkDelta(content=token)
                                )
                            ]
                        )
                        yield {"data": chunk.model_dump_json()}

                    # Send final chunk with session_id
                    final_chunk = ChatCompletionChunk(
                        id=request_id,
                        created=created_timestamp,
                        model=model,
                        choices=[
                            ChatCompletionChunkChoice(
                                delta=ChatCompletionChunkDelta(),
                                finish_reason="stop"
                            )
                        ],
                        x_session_id=session_id
                    )
                    yield {"data": final_chunk.model_dump_json()}
                    yield {"data": "[DONE]"}

                    # Save conversation
                    history.append({"role": "assistant", "content": assistant_message})
                    conversation_store.save_conversation(session_id, history)
                    db.update_session_message_count(session_id, len(history))

                except Exception as e:
                    error_data = {"error": {"message": str(e), "type": "internal_error"}}
                    yield {"data": json.dumps(error_data)}

            return EventSourceResponse(generate_stream())

        else:
            # Non-streaming response - Use agent system
            assistant_message = agent.run(user_input, conversation_history)

            # Save conversation
            history.append({"role": "assistant", "content": assistant_message})
            conversation_store.save_conversation(session_id, history)
            db.update_session_message_count(session_id, len(history))

            # Return OpenAI-compatible response
            response = ChatCompletionResponse(
                id=request_id,
                created=created_timestamp,
                model=model,
                choices=[
                    ChatCompletionChoice(
                        message=ChatMessage(role="assistant", content=assistant_message)
                    )
                ],
                x_session_id=session_id
            )

            return response

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid messages JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))