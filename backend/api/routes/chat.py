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
from backend.utils.file_handler import save_uploaded_files, extract_file_metadata
from backend.utils.auth import get_optional_user
from backend.agents import ChatAgent, ReActAgent, PlanExecuteAgent, AutoAgent, UltraworkAgent
from fastapi import Depends
import config

router = APIRouter(prefix="/v1", tags=["chat"])


def _get_agent(agent_type: str, model: str, temperature: float):
    """
    Get agent instance based on type

    Args:
        agent_type: Type of agent (chat, react, plan_execute, ultrawork, auto)
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
        # When opencode mode is enabled, use ultrawork instead of plan_execute
        if config.PYTHON_EXECUTOR_MODE == "opencode":
            return UltraworkAgent(model, temperature)
        return PlanExecuteAgent(model, temperature)
    elif agent_type == "ultrawork":
        return UltraworkAgent(model, temperature)
    elif agent_type == "auto":
        return AutoAgent(model, temperature)
    else:
        # Default to chat
        return ChatAgent(model, temperature)


def _prepare_messages_with_files(
    messages: List[ChatMessage],
    file_paths: List[str]
) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Prepare messages and file metadata (without injecting full content)

    Args:
        messages: Chat messages
        file_paths: List of file paths to include

    Returns:
        Tuple of (message_dicts, file_metadata)
    """
    from pathlib import Path

    # Convert Pydantic models to dicts
    message_dicts = [{"role": msg.role, "content": msg.content} for msg in messages]

    # If files are provided, create metadata (don't inject content)
    file_metadata = []
    if file_paths:
        for file_path in file_paths:
            path = Path(file_path)
            try:
                file_size = path.stat().st_size
                file_type = path.suffix.lstrip('.')

                # Determine file category
                text_extensions = ['txt', 'md', 'json', 'csv', 'py', 'js', 'html', 'xml', 'java', 'cpp', 'c', 'h', 'go', 'rs', 'ts', 'jsx', 'tsx']
                data_extensions = ['csv', 'xlsx', 'xls', 'json']
                code_extensions = ['py', 'js', 'java', 'cpp', 'c', 'h', 'go', 'rs', 'ts', 'jsx', 'tsx', 'html', 'css']

                category = 'binary'
                if file_type in text_extensions:
                    category = 'text'
                if file_type in data_extensions:
                    category = 'data'
                if file_type in code_extensions:
                    category = 'code'

                # Extract rich metadata based on file type
                rich_metadata = extract_file_metadata(file_path)

                file_metadata.append({
                    "name": path.name,
                    "path": file_path,
                    "size": file_size,
                    "type": file_type,
                    "category": category,
                    **rich_metadata  # Merge in the rich metadata
                })
            except Exception as e:
                file_metadata.append({
                    "name": path.name,
                    "path": file_path,
                    "error": str(e)
                })

    return message_dicts, file_metadata


@router.post("/chat/completions")
async def chat_completions(
    # Multipart form fields
    model: Optional[str] = Form(None),
    messages: str = Form(...),  # JSON string
    stream: str = Form("false"),
    temperature: Optional[str] = Form(None),
    max_tokens: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    agent_type: Optional[str] = Form(None),
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
        model: Optional model name (defaults to config.OLLAMA_MODEL)
        messages: JSON string of message list
        stream: "true" or "false" for streaming
        temperature: Optional temperature override
        max_tokens: Optional max tokens override
        session_id: Optional session ID to continue conversation
        agent_type: Optional agent type (defaults to config.DEFAULT_AGENT: chat, auto, react, plan_execute)
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
        
        # Use defaults from config if not specified
        model_name = model or config.OLLAMA_MODEL
        agent_type_name = agent_type or config.DEFAULT_AGENT

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
        file_metadata = []
        print(f"\n[CHAT] File upload check:")
        print(f"  files parameter: {files}")
        print(f"  files is None: {files is None}")
        print(f"  files length: {len(files) if files else 0}")
        if files and len(files) > 0:
            print(f"  ✓ Files detected, calling save_uploaded_files()...")
            file_paths = save_uploaded_files(files, username, session_id)
            print(f"  ✓ Files saved. Paths returned: {len(file_paths)}")
        else:
            print(f"  ✗ No files to save")

        # Prepare messages and file metadata (without full content injection)
        llm_messages, file_metadata = _prepare_messages_with_files(chat_messages, file_paths)

        # Get the appropriate agent
        agent = _get_agent(agent_type_name, model_name, temp)

        # Set session_id and username on agent for tool auth
        agent.session_id = session_id
        agent.username = username

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
                    for token in llm_backend.chat_stream(stream_messages, model_name, temp, session_id=session_id, agent_type="stream"):
                        assistant_message += token

                        # Send SSE chunk
                        chunk = ChatCompletionChunk(
                            id=request_id,
                            created=created_timestamp,
                            model=model_name,
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
                        model=model_name,
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
            # Non-streaming response - Use agent system with file metadata
            assistant_message = agent.run(user_input, conversation_history, file_metadata)

            # Save conversation
            history.append({"role": "assistant", "content": assistant_message})
            conversation_store.save_conversation(session_id, history)
            db.update_session_message_count(session_id, len(history))

            # Return OpenAI-compatible response
            response = ChatCompletionResponse(
                id=request_id,
                created=created_timestamp,
                model=model_name,
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