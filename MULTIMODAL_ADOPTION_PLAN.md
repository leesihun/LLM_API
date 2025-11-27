# Comprehensive Multimodal Adoption Plan

**Project:** LLM_API - AI-Powered Agentic Workflow System
**Version:** 2.0.0
**Date:** 2025-01-27
**Author:** System Architecture Analysis

---

## 📑 Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [Multimodal Adoption Plan](#multimodal-adoption-plan)
4. [Implementation Roadmap](#implementation-roadmap)
5. [Technical Decisions](#technical-decisions)
6. [Expected Benefits](#expected-benefits)

---

## 🎯 Executive Summary

This document provides a comprehensive plan for adopting **multimodal capabilities** (especially image inputs) for vision-enabled models like Llama 3.2 Vision, Gemma 2 Vision, and similar multimodal LLMs.

After thorough analysis of the codebase, this plan presents a **systematic, low-risk adoption path** that:
- Preserves existing text-based architecture
- Enables seamless vision model integration
- Maintains backward compatibility
- Follows the modular v2.0.0 architecture patterns

---

## 🔍 Current State Analysis

### ✅ **What Already Works Well**

#### 1. **File Upload System** (`backend/api/routes/files.py:28-75`)
- ✓ Handles multiple file uploads via multipart/form-data
- ✓ Saves files with unique IDs to user-specific directories
- ✓ Tracks file metadata (filename, size, content_type)

```python
@files_router.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    # Supports multiple files
    # Tracks metadata
    # User-specific storage
```

#### 2. **File Handling in Chat** (`backend/api/routes/chat.py:87-163`)
- ✓ `_handle_file_uploads()` function manages file uploads
- ✓ Files are passed to agents via `file_paths` parameter
- ✓ Session-based file persistence (reuses files across conversation turns)

```python
async def _handle_file_uploads(
    files: Optional[List[UploadFile]],
    session_id: Optional[str],
    user_id: str
) -> tuple[List[str], bool]:
    # Already supports file uploads
    # Retrieves old files from session history
    # Returns file paths for agent use
```

#### 3. **Modular Architecture** (v2.0.0)
- ✓ Clean separation: tools, agents, prompts, services
- ✓ LLMFactory for centralized LLM creation (`backend/utils/llm_factory.py:366-635`)
- ✓ Extensible tool system with BaseTool interface

```python
class LLMFactory:
    @classmethod
    def create_llm(cls, model, temperature, ...) -> ChatOllama:
        # Centralized LLM creation

    @classmethod
    def create_coder_llm(cls, ...) -> ChatOllama:
        # Specialized for code generation

    @classmethod
    def create_classifier_llm(cls, ...) -> ChatOllama:
        # Specialized for classification
```

### ❌ **Current Limitations**

#### 1. **Text-Only LLM Interaction**
- ✗ LangChain messages only contain text content
- ✗ No image encoding or multimodal message construction
- ✗ File metadata extracted as TEXT descriptions, not visual embeddings

```python
# Current: Text-only messages
conversation.append(HumanMessage(content=user_message))

# Needed: Multimodal messages
conversation.append(HumanMessage(content=[
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
    {"type": "text", "text": user_message}
]))
```

#### 2. **No Vision Model Support**
- ✗ Current models: `qwen3-coder:30b`, `gpt-oss:20b` (text-only)
- ✗ LLMFactory doesn't support vision-enabled models
- ✗ No image preprocessing or base64 encoding

```bash
# Current Ollama models (text-only)
$ curl http://127.0.0.1:11434/api/tags
{
  "models": [
    {"name": "qwen3-coder:30b", ...},
    {"name": "gpt-oss:20b", ...}
  ]
}

# Needed: Vision models
# - llama3.2-vision:11b
# - gemma-2-vision:9b
```

#### 3. **File Analyzer Limitation**
- ✗ Images analyzed via PIL metadata ONLY (`backend/tools/file_analyzer/`)
- ✗ No visual understanding (width, height, format only)
- ✗ No OCR, scene detection, or content analysis

```python
# Current: Metadata extraction only
if file_ext in {'.png', '.jpg', ...}:
    from PIL import Image
    with Image.open(file_path) as img:
        width, height = img.size
        mode = img.mode
    # Returns: width, height, format - NO visual understanding
```

#### 4. **Prompt Construction**
- ✗ Prompts are purely text-based
- ✗ No multimodal prompt templates
- ✗ File context is textual descriptions, not visual data

```python
# Current: Text-based file context
file_context = f"""
File: image.png
Type: PNG image
Size: 1920x1080
Format: RGB
"""

# Needed: Visual context via LLM vision
file_context = f"""
File: image.png
Visual Content: A bar chart showing sales data from 2020-2024.
Contains: Red bars for Q1, blue bars for Q2, title "Annual Sales"
Text Extracted (OCR): "Total Revenue: $1.2M"
"""
```

---

## 📋 Multimodal Adoption Plan

### **Phase 1: Infrastructure & Model Support** (Foundation)

#### **1.1 Update Ollama Model Configuration**

**Files to modify:**
- `backend/config/settings.py:34-36`

**Changes:**

```python
class Settings(BaseSettings):
    # ============================================================================
    # Ollama Configuration - MULTIMODAL SUPPORT
    # ============================================================================

    # Existing text models
    ollama_model: str = 'qwen3-coder:30b'
    agentic_classifier_model: str = 'qwen3-coder:30b'
    ollama_coder_model: str = 'qwen3-coder:30b'

    # NEW: Vision-capable models
    ollama_vision_model: str = 'llama3.2-vision:11b'  # or 'gemma-2-vision:9b'
    ollama_vision_enabled: bool = True
    ollama_vision_temperature: float = 0.3

    # NEW: Model capabilities registry
    # Tracks which models support which modalities
    model_capabilities: Dict[str, List[str]] = {
        'llama3.2-vision:11b': ['text', 'vision'],
        'gemma-2-vision:9b': ['text', 'vision'],
        'qwen3-coder:30b': ['text'],
        'gpt-oss:20b': ['text'],
        'deepseek-r1:1.5b': ['text']
    }
```

**Rationale:**
- Centralized model capability tracking
- Automatic vision detection based on model name
- Easy to extend for future modalities (audio, video)

**Installation Steps:**

```bash
# Pull vision models from Ollama
ollama pull llama3.2-vision:11b
ollama pull gemma-2-vision:9b

# Verify installation
ollama list
```

---

#### **1.2 Extend LLMFactory for Vision Models**

**Files to modify:**
- `backend/utils/llm_factory.py:366-635`

**New Methods:**

```python
class LLMFactory:
    """
    Factory class for creating and managing LLM instances.

    EXTENDED with vision model support.
    """

    @classmethod
    def create_vision_llm(
        cls,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = None,
        user_id: Optional[str] = None,
        enable_prompt_logging: bool = True,
        log_format: LogFormat = LogFormat.STRUCTURED,
        **kwargs
    ) -> ChatOllama:
        """
        Create a vision-capable LLM for multimodal tasks.

        Configures model to accept image inputs alongside text.
        Uses settings.ollama_vision_model by default.

        Args:
            model: Vision model name (defaults to settings.ollama_vision_model)
            temperature: Sampling temperature (defaults to settings.ollama_vision_temperature)
            num_ctx: Context window size (defaults to settings.ollama_num_ctx)
            user_id: User ID for prompt logging
            enable_prompt_logging: Enable prompt interception
            log_format: Format for prompt logs
            **kwargs: Additional parameters

        Returns:
            Configured ChatOllama instance for vision tasks

        Raises:
            ValueError: If specified model does not support vision

        Example:
            >>> vision_llm = LLMFactory.create_vision_llm(user_id="alice")
            >>> response = vision_llm.invoke([
            ...     HumanMessage(content=[
            ...         {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
            ...         {"type": "text", "text": "What's in this image?"}
            ...     ])
            ... ])
        """
        vision_model = model or settings.ollama_vision_model
        vision_temperature = (
            temperature if temperature is not None else settings.ollama_vision_temperature
        )

        # Verify model supports vision
        if not cls.supports_vision(vision_model):
            raise ValueError(
                f"Model '{vision_model}' does not support vision. "
                f"Available vision models: {cls.get_vision_models()}"
            )

        logger.debug(f"[LLMFactory] Creating vision LLM with model={vision_model}")

        return cls.create_llm(
            model=vision_model,
            temperature=vision_temperature,
            num_ctx=num_ctx,
            user_id=user_id,
            enable_prompt_logging=enable_prompt_logging,
            log_format=log_format,
            **kwargs
        )

    @classmethod
    def supports_vision(cls, model: str) -> bool:
        """
        Check if a model supports vision capabilities.

        Args:
            model: Model name to check

        Returns:
            True if model supports vision, False otherwise

        Example:
            >>> LLMFactory.supports_vision("llama3.2-vision:11b")
            True
            >>> LLMFactory.supports_vision("qwen3-coder:30b")
            False
        """
        capabilities = settings.model_capabilities.get(model, [])
        return 'vision' in capabilities

    @classmethod
    def get_vision_models(cls) -> List[str]:
        """
        Get list of all vision-capable models.

        Returns:
            List of model names that support vision

        Example:
            >>> LLMFactory.get_vision_models()
            ['llama3.2-vision:11b', 'gemma-2-vision:9b']
        """
        vision_models = []
        for model, capabilities in settings.model_capabilities.items():
            if 'vision' in capabilities:
                vision_models.append(model)
        return vision_models

    @classmethod
    def auto_select_llm(
        cls,
        has_images: bool = False,
        task_type: str = "general",
        user_id: Optional[str] = None,
        **kwargs
    ) -> ChatOllama:
        """
        Automatically select appropriate LLM based on input modality and task.

        Args:
            has_images: Whether input contains images
            task_type: Type of task (general, code, classification)
            user_id: User ID for prompt logging
            **kwargs: Additional parameters

        Returns:
            Appropriate LLM instance

        Example:
            >>> # Text-only task
            >>> llm = LLMFactory.auto_select_llm(has_images=False)

            >>> # Multimodal task with images
            >>> llm = LLMFactory.auto_select_llm(has_images=True)

            >>> # Code generation with images
            >>> llm = LLMFactory.auto_select_llm(has_images=True, task_type="code")
        """
        if has_images:
            # Use vision LLM for multimodal inputs
            logger.info("[LLMFactory] Auto-selecting vision LLM (images detected)")
            return cls.create_vision_llm(user_id=user_id, **kwargs)
        elif task_type == "code":
            # Use specialized coder LLM
            logger.info("[LLMFactory] Auto-selecting coder LLM")
            return cls.create_coder_llm(user_id=user_id, **kwargs)
        elif task_type == "classification":
            # Use classifier LLM
            logger.info("[LLMFactory] Auto-selecting classifier LLM")
            return cls.create_classifier_llm(user_id=user_id, **kwargs)
        else:
            # Default general-purpose LLM
            logger.info("[LLMFactory] Auto-selecting general LLM")
            return cls.create_llm(user_id=user_id, **kwargs)
```

**Rationale:**
- Type-safe vision model creation with capability validation
- Automatic model selection based on input modality
- Consistent interface with existing LLM creation methods
- Easy error handling for unsupported models

---

### **Phase 2: Message Construction & Image Encoding**

#### **2.1 Create Multimodal Message Builder**

**New file:** `backend/utils/multimodal_message.py`

```python
"""
Multimodal Message Construction for LangChain
==============================================
Handles image encoding and multimodal message formatting for Ollama vision models.

Features:
- Base64 image encoding
- MIME type detection
- Multimodal message construction
- File type separation (images vs data files)

Version: 1.0.0
Created: 2025-01-27
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import base64
import mimetypes
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class MultimodalMessageBuilder:
    """
    Builds LangChain messages with image content support for vision models.

    Supports Ollama's multimodal format:
    - Text-only messages
    - Image-only messages
    - Combined text + images
    - Multiple images per message

    Example:
        >>> builder = MultimodalMessageBuilder()
        >>> message = builder.build_multimodal_message(
        ...     text="What's in this image?",
        ...     image_paths=["chart.png"]
        ... )
        >>> # message.content = [
        >>> #     {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
        >>> #     {"type": "text", "text": "What's in this image?"}
        >>> # ]
    """

    # Supported image formats
    SUPPORTED_IMAGE_TYPES = {
        '.jpg', '.jpeg', '.png', '.gif', '.bmp',
        '.webp', '.tiff', '.tif', '.svg'
    }

    @staticmethod
    def encode_image(image_path: str) -> Dict[str, Any]:
        """
        Encode image to base64 for Ollama vision models.

        Follows Ollama's multimodal message format:
        https://github.com/ollama/ollama/blob/main/docs/api.md#chat-request-with-images

        Args:
            image_path: Path to image file

        Returns:
            Dict with type='image_url' and image_url data

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If file type is not supported

        Example:
            >>> encoded = MultimodalMessageBuilder.encode_image("photo.jpg")
            >>> encoded
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
                }
            }
        """
        path = Path(image_path)

        # Validate file existence
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Validate file type
        if path.suffix.lower() not in MultimodalMessageBuilder.SUPPORTED_IMAGE_TYPES:
            raise ValueError(
                f"Unsupported image type: {path.suffix}. "
                f"Supported types: {MultimodalMessageBuilder.SUPPORTED_IMAGE_TYPES}"
            )

        # Read and encode to base64
        try:
            with open(path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"[MultimodalMessage] Failed to read image {image_path}: {e}")
            raise

        # Detect MIME type
        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type or not mime_type.startswith('image/'):
            # Fallback based on extension
            mime_map = {
                '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                '.png': 'image/png', '.gif': 'image/gif',
                '.bmp': 'image/bmp', '.webp': 'image/webp',
                '.tiff': 'image/tiff', '.tif': 'image/tiff',
                '.svg': 'image/svg+xml'
            }
            mime_type = mime_map.get(path.suffix.lower(), 'image/jpeg')

        logger.debug(f"[MultimodalMessage] Encoded {path.name} ({mime_type}, {len(image_data)} chars)")

        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{image_data}"
            }
        }

    @staticmethod
    def build_multimodal_message(
        text: str,
        image_paths: Optional[List[str]] = None,
        role: str = "user"
    ) -> Union[HumanMessage, AIMessage, SystemMessage]:
        """
        Build a multimodal message with text and optional images.

        Images are added BEFORE text in the content array, as recommended
        by Ollama for optimal vision model performance.

        Args:
            text: Text content
            image_paths: List of image file paths (optional)
            role: Message role (user/assistant/system)

        Returns:
            LangChain message with multimodal content

        Raises:
            ValueError: If role is invalid

        Example:
            >>> # Text + single image
            >>> msg = MultimodalMessageBuilder.build_multimodal_message(
            ...     text="Analyze this chart",
            ...     image_paths=["chart.png"]
            ... )

            >>> # Text + multiple images
            >>> msg = MultimodalMessageBuilder.build_multimodal_message(
            ...     text="Compare these images",
            ...     image_paths=["before.png", "after.png"]
            ... )

            >>> # Text only (backwards compatible)
            >>> msg = MultimodalMessageBuilder.build_multimodal_message(
            ...     text="Hello, world!"
            ... )
        """
        content = []

        # Add images first (recommended by Ollama)
        if image_paths:
            for img_path in image_paths:
                try:
                    encoded = MultimodalMessageBuilder.encode_image(img_path)
                    content.append(encoded)
                    logger.info(f"[MultimodalMessage] Added image: {Path(img_path).name}")
                except Exception as e:
                    logger.warning(f"[MultimodalMessage] Failed to encode {img_path}: {e}")
                    # Continue with other images - don't fail entire message

        # Add text content
        if text:
            content.append({
                "type": "text",
                "text": text
            })

        # If only text, use simple string format (backwards compatible)
        if not image_paths and text:
            content = text

        # Build message based on role
        if role == "user":
            return HumanMessage(content=content)
        elif role == "assistant":
            return AIMessage(content=content)
        elif role == "system":
            return SystemMessage(content=content)
        else:
            raise ValueError(f"Invalid role: {role}. Must be 'user', 'assistant', or 'system'")

    @staticmethod
    def is_image_file(file_path: str) -> bool:
        """
        Check if file is a supported image format.

        Args:
            file_path: Path to file

        Returns:
            True if file is a supported image, False otherwise

        Example:
            >>> MultimodalMessageBuilder.is_image_file("photo.jpg")
            True
            >>> MultimodalMessageBuilder.is_image_file("data.csv")
            False
        """
        return Path(file_path).suffix.lower() in MultimodalMessageBuilder.SUPPORTED_IMAGE_TYPES

    @staticmethod
    def separate_files_by_type(file_paths: List[str]) -> Dict[str, List[str]]:
        """
        Separate files into images and non-images (data files).

        This is useful for routing files to appropriate tools:
        - Images → vision_analyzer or vision LLM
        - Data files → python_coder or file_analyzer

        Args:
            file_paths: List of file paths

        Returns:
            Dict with 'images' and 'data_files' lists

        Example:
            >>> files = ["chart.png", "data.csv", "photo.jpg", "report.xlsx"]
            >>> separated = MultimodalMessageBuilder.separate_files_by_type(files)
            >>> separated
            {
                'images': ['chart.png', 'photo.jpg'],
                'data_files': ['data.csv', 'report.xlsx']
            }
        """
        images = []
        data_files = []

        for path in file_paths:
            if MultimodalMessageBuilder.is_image_file(path):
                images.append(path)
            else:
                data_files.append(path)

        logger.debug(
            f"[MultimodalMessage] Separated {len(file_paths)} files: "
            f"{len(images)} images, {len(data_files)} data files"
        )

        return {
            'images': images,
            'data_files': data_files
        }

    @staticmethod
    def get_image_count(file_paths: List[str]) -> int:
        """
        Count number of image files in a list.

        Args:
            file_paths: List of file paths

        Returns:
            Number of image files

        Example:
            >>> files = ["chart.png", "data.csv", "photo.jpg"]
            >>> MultimodalMessageBuilder.get_image_count(files)
            2
        """
        return len([f for f in file_paths if MultimodalMessageBuilder.is_image_file(f)])


# Convenience functions for backward compatibility

def encode_image(image_path: str) -> Dict[str, Any]:
    """Encode image to base64. Convenience wrapper."""
    return MultimodalMessageBuilder.encode_image(image_path)


def is_image(file_path: str) -> bool:
    """Check if file is an image. Convenience wrapper."""
    return MultimodalMessageBuilder.is_image_file(file_path)


def separate_images(file_paths: List[str]) -> Dict[str, List[str]]:
    """Separate images from data files. Convenience wrapper."""
    return MultimodalMessageBuilder.separate_files_by_type(file_paths)
```

**Rationale:**
- Centralized multimodal message construction
- Base64 encoding with MIME type detection
- Backwards compatible with text-only workflows
- Automatic file type separation for routing
- Comprehensive error handling and logging

---

#### **2.2 Update Chat Task for Multimodal Support**

**Files to modify:**
- `backend/tasks/chat_task.py:14-86`

**Updated Implementation:**

```python
"""
Normal Chat Task
Simple chat with or without conversation memory

UPDATED: Now supports multimodal inputs (text + images)
"""

from typing import List, Optional, AsyncIterator
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from backend.config.settings import settings
from backend.models.schemas import ChatMessage
from backend.storage.conversation_store import conversation_store
from backend.utils.multimodal_message import MultimodalMessageBuilder


class ChatTask:
    """
    Handles normal chat conversations.

    ENHANCED with multimodal support:
    - Automatically switches to vision LLM when images detected
    - Maintains backward compatibility for text-only chat
    """

    def __init__(self):
        from backend.utils.llm_factory import LLMFactory
        self.LLMFactory = LLMFactory
        self.llm = None
        self.current_user_id = None
        self.current_is_vision = False  # Track if using vision LLM

    async def execute(
        self,
        messages: List[ChatMessage],
        session_id: Optional[str] = None,
        use_memory: bool = True,
        user_id: str = "default",
        file_paths: Optional[List[str]] = None  # NEW: Support file attachments
    ) -> str:
        """
        Execute a simple chat interaction with optional multimodal inputs.

        Args:
            messages: List of chat messages (current conversation)
            session_id: Optional session ID for conversation history
            use_memory: Whether to use conversation memory
            user_id: User ID for prompt logging
            file_paths: Optional list of file paths (images will be encoded)

        Returns:
            AI response text
        """
        # Separate images from data files
        file_types = MultimodalMessageBuilder.separate_files_by_type(file_paths or [])
        has_images = len(file_types['images']) > 0

        # Auto-select appropriate LLM based on input modality
        needs_vision = has_images
        needs_refresh = (
            self.llm is None or
            self.current_user_id != user_id or
            self.current_is_vision != needs_vision  # Switch LLM if modality changed
        )

        if needs_refresh:
            if needs_vision:
                self.llm = self.LLMFactory.create_vision_llm(user_id=user_id)
                self.current_is_vision = True
                logger.info(f"[ChatTask] Using vision LLM for multimodal input ({len(file_types['images'])} images)")
            else:
                self.llm = self.LLMFactory.create_llm(user_id=user_id)
                self.current_is_vision = False
                logger.info(f"[ChatTask] Using text LLM")

            self.current_user_id = user_id

        # Build conversation with multimodal support
        conversation = self._build_multimodal_conversation(
            messages,
            session_id,
            use_memory,
            image_paths=file_types['images']
        )

        # Generate response
        response = await self.llm.ainvoke(conversation)

        return response.content

    def _build_multimodal_conversation(
        self,
        messages: List[ChatMessage],
        session_id: Optional[str] = None,
        use_memory: bool = True,
        image_paths: Optional[List[str]] = None
    ) -> List:
        """
        Build conversation context with multimodal message support.

        Images are attached ONLY to the last user message to avoid
        redundant encoding and token usage.

        Args:
            messages: Current conversation messages
            session_id: Session ID for history retrieval
            use_memory: Whether to include conversation history
            image_paths: List of image paths (attached to last user message)

        Returns:
            List of LangChain messages (HumanMessage, AIMessage, SystemMessage)
        """
        conversation = []

        # Add conversation history (text-only for now)
        if use_memory and session_id:
            history = conversation_store.get_messages(session_id, limit=1000)

            for msg in history:
                if msg.role == "user":
                    conversation.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    conversation.append(AIMessage(content=msg.content))

        # Add current messages (potentially multimodal)
        for i, msg in enumerate(messages):
            is_last_user_message = (i == len(messages) - 1 and msg.role == "user")

            # Attach images ONLY to the last user message
            images_for_msg = image_paths if is_last_user_message else None

            if images_for_msg:
                # Build multimodal message
                logger.info(f"[ChatTask] Building multimodal message with {len(images_for_msg)} image(s)")
                mm_message = MultimodalMessageBuilder.build_multimodal_message(
                    text=msg.content,
                    image_paths=images_for_msg,
                    role=msg.role
                )
                conversation.append(mm_message)
            else:
                # Standard text-only message
                if msg.role == "user":
                    conversation.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    conversation.append(AIMessage(content=msg.content))

        logger.debug(f"[ChatTask] Built conversation with {len(conversation)} messages")
        return conversation


# Global chat task instance
chat_task = ChatTask()
```

**Rationale:**
- Seamless text/vision LLM switching based on input type
- Automatic multimodal message construction
- Images attached only to last user message (efficiency)
- Backward compatible with text-only workflows
- Conversation history preserved as text (images only in current turn)

---

### **Phase 3: Tool-Level Multimodal Support**

#### **3.1 Enhance File Analyzer with Vision**

**Files to modify:**
- `backend/tools/file_analyzer/tool.py:23-191`

**New Vision Analysis Capability:**

```python
"""
File Analyzer Tool - BaseTool Implementation
=============================================
ENHANCED with vision-based image analysis.

Version: 2.1.0 (Multimodal)
"""

from typing import List, Dict, Any, Optional

from backend.core import BaseTool, ToolResult, FileAnalysisResult
from backend.utils.logging_utils import get_logger
from backend.tools.file_analyzer.analyzer import FileAnalyzer as FileAnalyzerCore
from backend.services.file_handler import file_handler_registry
from backend.utils.llm_factory import LLMFactory
from backend.utils.multimodal_message import MultimodalMessageBuilder

logger = get_logger(__name__)


class FileAnalyzer(BaseTool):
    """
    File analysis tool with BaseTool interface.

    ENHANCED Features:
    - Comprehensive file analysis for multiple formats
    - VISION-BASED image analysis (scene, OCR, objects)
    - Uses unified FileHandlerRegistry
    - Extracts metadata, statistics, and previews
    - Returns standardized ToolResult

    Supported formats: CSV, Excel, JSON, Text, PDF, DOCX, Images

    Usage:
        >>> tool = FileAnalyzer(use_vision=True)
        >>> result = await tool.execute(
        ...     query="What does this chart show?",
        ...     file_paths=["chart.png"]
        ... )
        >>> print(result.output)
    """

    def __init__(self, use_llm_for_complex: bool = False, use_vision: bool = True):
        """
        Initialize File Analyzer Tool.

        Args:
            use_llm_for_complex: Use LLM for complex file analysis
            use_vision: Enable vision-based image analysis (default: True)
        """
        super().__init__()

        # Initialize core analyzer
        self.analyzer = FileAnalyzerCore(use_llm_for_complex=use_llm_for_complex)

        # Also have direct access to unified registry
        self.file_registry = file_handler_registry

        # NEW: Initialize vision LLM if enabled
        self.use_vision = use_vision
        self.vision_llm = None
        if use_vision:
            try:
                self.vision_llm = LLMFactory.create_vision_llm()
                logger.info("[FileAnalyzer] Vision-based image analysis ENABLED")
            except Exception as e:
                logger.warning(f"[FileAnalyzer] Failed to load vision LLM: {e}. Vision analysis DISABLED.")
                self.use_vision = False
        else:
            logger.info("[FileAnalyzer] Vision-based image analysis DISABLED")

    async def execute(
        self,
        query: str,
        context: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        user_query: str = "",
        **kwargs
    ) -> ToolResult:
        """
        Execute file analysis with vision support.

        For images:
        - If use_vision=True: Deep visual analysis (scene, OCR, objects)
        - If use_vision=False: Basic metadata only (width, height, format)

        Args:
            query: User's question or analysis request
            context: Optional additional context
            file_paths: List of file paths to analyze
            user_query: User's original question (for context)
            **kwargs: Additional parameters

        Returns:
            ToolResult with analysis results
        """
        self._log_execution_start(
            query=query[:100],
            file_count=len(file_paths) if file_paths else 0
        )

        try:
            # Validate inputs
            if not self.validate_inputs(file_paths=file_paths):
                return self._handle_validation_error(
                    "file_paths must be a non-empty list",
                    parameter="file_paths"
                )

            # Use user_query if provided, otherwise use query
            analysis_query = user_query or query

            # Separate images from data files
            file_types = MultimodalMessageBuilder.separate_files_by_type(file_paths)
            images = file_types['images']
            data_files = file_types['data_files']

            # Analyze data files with core analyzer
            data_results = []
            if data_files:
                result_dict = self.analyzer.analyze(
                    file_paths=data_files,
                    user_query=analysis_query
                )
                data_results = result_dict.get('results', [])

            # Analyze images with vision (if enabled)
            image_results = []
            if images and self.use_vision and self.vision_llm:
                logger.info(f"[FileAnalyzer] Analyzing {len(images)} image(s) with vision")
                for img_path in images:
                    vision_result = await self._analyze_image_with_vision(img_path, analysis_query)
                    image_results.append(vision_result)
            elif images:
                # Fallback: basic metadata only
                logger.info(f"[FileAnalyzer] Analyzing {len(images)} image(s) with metadata only")
                for img_path in images:
                    metadata_result = self.analyzer.analyze(
                        file_paths=[img_path],
                        user_query=analysis_query
                    )
                    image_results.extend(metadata_result.get('results', []))

            # Combine results
            all_results = data_results + image_results

            # Build combined result dict
            combined_result = {
                'success': True,
                'results': all_results,
                'files_analyzed': len(all_results),
                'summary': self._build_combined_summary(data_results, image_results)
            }

            # Convert to ToolResult
            tool_result = self._convert_to_tool_result(combined_result)

            self._log_execution_end(tool_result)
            return tool_result

        except Exception as e:
            return self._handle_error(e, "execute")

    async def _analyze_image_with_vision(
        self,
        image_path: str,
        user_query: str
    ) -> Dict[str, Any]:
        """
        Analyze image using vision LLM for deep understanding.

        Goes beyond metadata to provide:
        - Scene description (what's in the image)
        - OCR text extraction (any visible text)
        - Object detection (key objects and relationships)
        - Content analysis (insights relevant to user's question)

        Args:
            image_path: Path to image file
            user_query: User's question about the image

        Returns:
            Dict with vision analysis results
        """
        from pathlib import Path
        import json

        prompt = f"""Analyze this image in detail for the following task:

USER QUESTION: {user_query}

Provide a comprehensive analysis in JSON format with these fields:

{{
  "scene_description": "Detailed description of what's in the image",
  "text_found": ["List", "of", "extracted", "text", "via", "OCR"],
  "objects_detected": ["List", "of", "key", "objects"],
  "chart_type": "If this is a chart/graph, identify the type (bar, line, pie, etc.)",
  "key_insights": "Insights relevant to the user's question",
  "data_extracted": "If chart/table, extract key data points"
}}

Focus on information that helps answer the user's question.
"""

        try:
            # Build multimodal message
            message = MultimodalMessageBuilder.build_multimodal_message(
                text=prompt,
                image_paths=[image_path]
            )

            # Get vision LLM response
            response = await self.vision_llm.ainvoke([message])

            # Parse vision analysis
            vision_data = self._parse_vision_response(response.content)

            # Build result in standard format
            result = {
                'file': Path(image_path).name,
                'success': True,
                'type': 'image',
                'analysis_method': 'vision_llm',
                'scene_description': vision_data.get('scene_description', ''),
                'text_found': vision_data.get('text_found', []),
                'objects_detected': vision_data.get('objects_detected', []),
                'chart_type': vision_data.get('chart_type', 'N/A'),
                'key_insights': vision_data.get('key_insights', ''),
                'data_extracted': vision_data.get('data_extracted', '')
            }

            logger.info(f"[FileAnalyzer] Vision analysis completed for {Path(image_path).name}")
            return result

        except Exception as e:
            logger.error(f"[FileAnalyzer] Vision analysis failed for {image_path}: {e}")
            return {
                'file': Path(image_path).name,
                'success': False,
                'error': f"Vision analysis failed: {str(e)}"
            }

    def _parse_vision_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse vision LLM response (expects JSON format).

        Args:
            response_text: Raw response from vision LLM

        Returns:
            Parsed vision data dict
        """
        import json

        try:
            # Extract JSON from markdown code blocks if present
            if "```json" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_text = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_text = response_text.strip()

            # Parse JSON
            data = json.loads(json_text)
            return data

        except json.JSONDecodeError as e:
            logger.warning(f"[FileAnalyzer] Failed to parse vision response as JSON: {e}")
            # Fallback: treat entire response as scene description
            return {
                'scene_description': response_text,
                'text_found': [],
                'objects_detected': [],
                'key_insights': response_text[:200]
            }

    def _build_combined_summary(
        self,
        data_results: List[Dict],
        image_results: List[Dict]
    ) -> str:
        """Build summary combining data file and image analysis."""
        parts = []

        if data_results:
            parts.append(f"Analyzed {len(data_results)} data file(s)")

        if image_results:
            vision_count = sum(1 for r in image_results if r.get('analysis_method') == 'vision_llm')
            if vision_count > 0:
                parts.append(f"Analyzed {vision_count} image(s) with vision AI")
            else:
                parts.append(f"Analyzed {len(image_results)} image(s) with metadata")

        return "; ".join(parts) if parts else "No files analyzed"

    # ... rest of existing methods (validate_inputs, _convert_to_tool_result, etc.) ...


# Global singleton instance with vision enabled by default
file_analyzer = FileAnalyzer(use_vision=True)
```

**Rationale:**
- Deep image understanding beyond basic metadata
- OCR for text extraction from screenshots/documents
- Scene description for visual context
- Chart/graph detection and data extraction
- Graceful fallback to metadata-only if vision unavailable

---

#### **3.2 Update Python Coder for Image Context**

**Files to modify:**
- `backend/tools/python_coder/orchestrator.py:693-741`

**Enhancement in `_prepare_files()` method:**

```python
def _prepare_files(
    self,
    file_paths: List[str]
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Prepare and validate input files with MULTIMODAL awareness.

    For images: Extracts visual metadata + optional vision descriptions
    For data files: Existing metadata extraction

    Args:
        file_paths: List of file paths

    Returns:
        Tuple of (validated_files, file_metadata)
    """
    from backend.utils.multimodal_message import MultimodalMessageBuilder

    validated_files = {}
    file_metadata = {}

    # Separate images from data files
    file_types = MultimodalMessageBuilder.separate_files_by_type(file_paths)

    # Process images with visual awareness
    for img_path in file_types['images']:
        path = Path(img_path)

        # Validate existence
        if not path.exists():
            logger.error(f"[PythonCoderTool] File not found: {img_path}")
            continue

        # Validate size
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > settings.python_code_max_file_size:
            logger.error(f"[PythonCoderTool] File too large: {img_path} ({size_mb:.2f}MB)")
            continue

        # Extract basic metadata
        metadata = self.file_handler_factory.extract_metadata(path)

        # Add image-specific context
        metadata['file_category'] = 'image'
        metadata['visual_context'] = (
            f"Image file that can be loaded with PIL/matplotlib. "
            f"Size: {metadata.get('width', '?')}x{metadata.get('height', '?')} pixels. "
            f"Format: {metadata.get('format', 'unknown')}. "
            f"Suggested usage: Use PIL.Image.open('{get_original_filename(path.name)}') "
            f"or matplotlib.pyplot.imread() for analysis/display."
        )

        # Extract ORIGINAL filename
        original_filename = get_original_filename(path.name)
        metadata['original_filename'] = original_filename

        file_metadata[str(path)] = metadata
        validated_files[str(path)] = original_filename

        logger.info(f"[PythonCoderTool] Validated image: {original_filename} ({size_mb:.2f}MB)")

    # Process data files with existing logic
    for data_path in file_types['data_files']:
        path = Path(data_path)

        # Validate existence
        if not path.exists():
            logger.error(f"[PythonCoderTool] File not found: {data_path}")
            continue

        # Validate size
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > settings.python_code_max_file_size:
            logger.error(f"[PythonCoderTool] File too large: {data_path} ({size_mb:.2f}MB)")
            continue

        # Validate type
        if not self.executor.validate_file_type(data_path):
            logger.error(f"[PythonCoderTool] Unsupported file type: {data_path}")
            continue

        # Extract ORIGINAL filename
        original_filename = get_original_filename(path.name)

        # Extract metadata using FileHandlerFactory
        metadata = self.file_handler_factory.extract_metadata(path)
        metadata['original_filename'] = original_filename
        metadata['file_category'] = 'data'
        file_metadata[str(path)] = metadata

        # Store mapping (temp path -> ORIGINAL filename for execution)
        validated_files[str(path)] = original_filename

        logger.info(f"[PythonCoderTool] Validated file: {original_filename} (temp: {path.name}, {size_mb:.2f}MB)")

    return validated_files, file_metadata
```

**Rationale:**
- Code generation can reference visual content contextually
- LLM receives image loading suggestions (PIL, matplotlib)
- Separation ensures images don't interfere with data file parsing
- Generated code can properly handle both image and data files

**Example Generated Code:**

```python
# With multimodal awareness, LLM generates:

from PIL import Image
import matplotlib.pyplot as plt

# Load the chart image
img = Image.open('sales_chart.png')

# Display the image
plt.figure(figsize=(10, 6))
plt.imshow(img)
plt.axis('off')
plt.title('Sales Chart Visualization')
plt.savefig('chart_display.png', dpi=150, bbox_inches='tight')

print("Chart displayed and saved!")
```

---

### **Phase 4: ReAct Agent Multimodal Integration**

#### **4.1 Add Vision Tool to ReAct**

**New file:** `backend/tools/vision_analyzer/tool.py`

```python
"""
Vision Analyzer Tool
====================
Provides deep image understanding capabilities to ReAct agent using vision LLMs.

This tool enables the agent to:
- Understand visual content (scenes, objects, layouts)
- Extract text from images (OCR)
- Analyze charts and diagrams
- Answer questions about visual content

Version: 1.0.0
Created: 2025-01-27
"""

from typing import Optional, List, Dict, Any
from pathlib import Path

from backend.core import BaseTool, ToolResult
from backend.utils.llm_factory import LLMFactory
from backend.utils.multimodal_message import MultimodalMessageBuilder
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class VisionAnalyzer(BaseTool):
    """
    Vision analysis tool for understanding images in ReAct workflows.

    This tool provides the ReAct agent with visual reasoning capabilities,
    allowing it to analyze images and answer questions about visual content.

    Capabilities:
    - Scene description (what's happening in the image)
    - OCR text extraction (visible text from screenshots, documents, signs)
    - Object detection (identify and locate objects)
    - Visual question answering (answer specific questions about images)
    - Chart analysis (extract data from graphs, tables, diagrams)

    Example Usage in ReAct:
        Thought: The user wants to know what's in the uploaded chart.
        Action: vision_analyzer
        Action Input: What data is shown in this chart? Extract the title and key values.
        Observation: [Chart shows quarterly sales from 2020-2024, with Q4 2024
                      reaching $1.2M, which is 15% higher than Q4 2023...]
    """

    # Analysis focus types
    FOCUS_GENERAL = "general"      # General visual analysis
    FOCUS_OCR = "ocr"             # Text extraction focus
    FOCUS_OBJECTS = "objects"      # Object detection focus
    FOCUS_SCENE = "scene"         # Scene understanding focus
    FOCUS_CHART = "chart"         # Chart/data visualization focus

    def __init__(self):
        """Initialize Vision Analyzer with vision-capable LLM."""
        super().__init__()

        try:
            self.vision_llm = LLMFactory.create_vision_llm()
            logger.info("[VisionAnalyzer] Initialized with vision LLM")
        except Exception as e:
            logger.error(f"[VisionAnalyzer] Failed to initialize vision LLM: {e}")
            raise

    async def execute(
        self,
        query: str,
        file_paths: Optional[List[str]] = None,
        focus: str = "general",
        context: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """
        Analyze images to answer user's visual question.

        Args:
            query: User's question about the image(s)
            file_paths: List of file paths (images will be filtered automatically)
            focus: Analysis focus - one of: general, ocr, objects, scene, chart
            context: Optional additional context about the task
            **kwargs: Additional parameters

        Returns:
            ToolResult with visual analysis

        Example:
            >>> tool = VisionAnalyzer()
            >>> result = await tool.execute(
            ...     query="What does this chart show?",
            ...     file_paths=["sales_chart.png"],
            ...     focus="chart"
            ... )
            >>> print(result.output)
        """
        self._log_execution_start(query=query[:100], focus=focus)

        try:
            # Validate inputs
            if not file_paths:
                return self._handle_validation_error(
                    "No files provided for vision analysis",
                    parameter="file_paths"
                )

            # Filter to images only
            file_types = MultimodalMessageBuilder.separate_files_by_type(file_paths)
            images = file_types['images']

            if not images:
                return self._handle_validation_error(
                    f"No image files found. Received: {[Path(f).suffix for f in file_paths]}",
                    parameter="file_paths"
                )

            logger.info(f"[VisionAnalyzer] Analyzing {len(images)} image(s) with focus: {focus}")

            # Build specialized vision prompt
            prompt = self._build_vision_prompt(query, focus, context)

            # Analyze all images
            results = []
            for img_path in images:
                try:
                    # Build multimodal message
                    message = MultimodalMessageBuilder.build_multimodal_message(
                        text=prompt,
                        image_paths=[img_path]
                    )

                    # Get vision analysis
                    response = await self.vision_llm.ainvoke([message])

                    results.append({
                        'image': Path(img_path).name,
                        'analysis': response.content,
                        'success': True
                    })

                    logger.info(f"[VisionAnalyzer] ✓ Analyzed {Path(img_path).name}")

                except Exception as e:
                    logger.error(f"[VisionAnalyzer] ✗ Failed to analyze {img_path}: {e}")
                    results.append({
                        'image': Path(img_path).name,
                        'error': str(e),
                        'success': False
                    })

            # Format output for ReAct agent
            output = self._format_vision_results(results, query, focus)

            # Build metadata
            metadata = {
                'images_analyzed': len(images),
                'focus': focus,
                'successful_analyses': sum(1 for r in results if r.get('success')),
                'failed_analyses': sum(1 for r in results if not r.get('success')),
                'results': results
            }

            tool_result = ToolResult(
                success=True,
                output=output,
                metadata=metadata
            )

            self._log_execution_end(tool_result)
            return tool_result

        except Exception as e:
            return self._handle_error(e, "execute")

    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate tool inputs.

        Args:
            **kwargs: Must contain 'file_paths' with at least one image file

        Returns:
            True if inputs are valid
        """
        file_paths = kwargs.get("file_paths")

        # File paths must be provided
        if not file_paths or not isinstance(file_paths, list):
            return False

        # At least one file must be an image
        has_images = any(
            MultimodalMessageBuilder.is_image_file(f)
            for f in file_paths
        )

        return has_images

    def _build_vision_prompt(
        self,
        query: str,
        focus: str,
        context: Optional[str] = None
    ) -> str:
        """
        Build specialized vision prompt based on analysis focus.

        Args:
            query: User's question
            focus: Analysis focus (general, ocr, objects, scene, chart)
            context: Optional additional context

        Returns:
            Formatted prompt for vision LLM
        """
        base = f"**User Question:** {query}\n\n"

        if context:
            base += f"**Context:** {context}\n\n"

        if focus == self.FOCUS_OCR:
            base += """**Task:** Extract ALL visible text from this image.

Instructions:
- Extract text exactly as shown (preserve formatting)
- Include text from labels, titles, axis labels, legends, annotations
- Note text positions if relevant (top, center, bottom, etc.)
- If no text is found, explicitly state "No text detected"

Output the extracted text clearly."""

        elif focus == self.FOCUS_OBJECTS:
            base += """**Task:** Identify and list all objects in this image.

Instructions:
- List all visible objects
- Note their positions and relationships
- Describe key visual attributes (colors, sizes, shapes)
- Identify the main subject vs background elements

Provide a structured list of objects."""

        elif focus == self.FOCUS_SCENE:
            base += """**Task:** Describe the overall scene in this image.

Instructions:
- Describe the setting and environment
- Identify the context and purpose of the image
- Note the mood, composition, and visual style
- Explain what's happening or what the image represents

Provide a comprehensive scene description."""

        elif focus == self.FOCUS_CHART:
            base += """**Task:** Analyze this chart/diagram/visualization.

Instructions:
- Identify the chart type (bar, line, pie, scatter, etc.)
- Extract the title and axis labels
- List key data points and values
- Note trends, patterns, or notable features
- If there's a legend, explain what each element represents

Provide structured data extraction and insights."""

        else:  # FOCUS_GENERAL
            base += """**Task:** Analyze this image to answer the user's question.

Instructions:
- Focus on information relevant to the question
- Be specific and detailed
- Extract any visible text if relevant
- Describe visual elements that help answer the question
- Provide actionable insights

Be comprehensive but concise."""

        return base

    def _format_vision_results(
        self,
        results: List[Dict[str, Any]],
        query: str,
        focus: str
    ) -> str:
        """
        Format vision analysis results for ReAct agent observation.

        Args:
            results: List of analysis results for each image
            query: Original user query
            focus: Analysis focus used

        Returns:
            Formatted output string for ReAct observation
        """
        output_parts = []

        # Header
        total_images = len(results)
        successful = sum(1 for r in results if r.get('success'))
        output_parts.append(
            f"Vision Analysis Results ({successful}/{total_images} images analyzed, focus: {focus}):\n"
        )

        # Individual image results
        for i, result in enumerate(results, 1):
            image_name = result.get('image', f'Image {i}')

            output_parts.append(f"\n{'='*60}")
            output_parts.append(f"Image {i}: {image_name}")
            output_parts.append('='*60)

            if result.get('success'):
                analysis = result.get('analysis', 'No analysis available')
                output_parts.append(analysis)
            else:
                error = result.get('error', 'Unknown error')
                output_parts.append(f"ERROR: {error}")

        # Footer
        output_parts.append(f"\n{'='*60}")
        output_parts.append("End of Vision Analysis")
        output_parts.append('='*60)

        return "\n".join(output_parts)


# Global singleton instance
vision_analyzer = VisionAnalyzer()


# Backward compatibility function
async def analyze_vision(
    query: str,
    file_paths: List[str],
    focus: str = "general"
) -> Dict[str, Any]:
    """
    Legacy function for backward compatibility.

    Args:
        query: User's question about the image
        file_paths: List of image paths
        focus: Analysis focus

    Returns:
        Vision analysis result dictionary
    """
    result = await vision_analyzer.execute(
        query=query,
        file_paths=file_paths,
        focus=focus
    )

    return {
        'success': result.success,
        'output': result.output,
        'metadata': result.metadata,
        'error': result.error
    }


# Export for backward compatibility
__all__ = [
    'VisionAnalyzer',
    'vision_analyzer',
    'analyze_vision',
]
```

**New file:** `backend/tools/vision_analyzer/__init__.py`

```python
"""
Vision Analyzer Tool
Exports the vision analyzer tool for ReAct agent integration
"""

from .tool import (
    VisionAnalyzer,
    vision_analyzer,
    analyze_vision,
)

__all__ = [
    'VisionAnalyzer',
    'vision_analyzer',
    'analyze_vision',
]
```

**Rationale:**
- Dedicated vision tool for ReAct's multimodal reasoning loop
- Multiple focus modes for different analysis tasks
- Clean integration with BaseTool interface
- Detailed prompts for optimal vision model performance

---

#### **4.2 Register Vision Tool in ReAct**

**Files to modify:**

1. **`backend/tasks/react/models.py`** - Add VISION_ANALYZER to ToolName enum

```python
"""
ReAct Agent Models
Data models and enums for ReAct agent
"""

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel

# ... existing imports ...


class ToolName(str, Enum):
    """
    Available tools for ReAct agent.

    UPDATED with vision_analyzer for multimodal support.
    """
    WEB_SEARCH = "web_search"
    RAG_RETRIEVAL = "rag_retrieval"
    PYTHON_CODER = "python_coder"
    FILE_ANALYZER = "file_analyzer"
    VISION_ANALYZER = "vision_analyzer"  # NEW: Vision-based image analysis
    FINISH = "finish"


# ... rest of existing code ...
```

2. **`backend/tasks/react/execution.py`** - Add vision tool routing in ToolExecutor

```python
"""
Tool Execution Module
Handles execution of tools selected by ReAct agent
"""

from typing import Optional, List, Dict, Any
from .models import ToolName
# ... existing imports ...

from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ToolExecutor:
    """
    Executes tools based on ReAct agent's action selection.

    UPDATED with vision_analyzer support.
    """

    def __init__(self, llm, user_id: str = "default"):
        self.llm = llm
        self.user_id = user_id

    async def execute(
        self,
        action: ToolName,
        action_input: str,
        file_paths: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        steps: Optional[List] = None
    ) -> str:
        """
        Execute the selected tool action.

        Args:
            action: Tool to execute
            action_input: Input/query for the tool
            file_paths: Optional file paths for file-based tools
            session_id: Session ID for stateful tools
            steps: Previous ReAct steps for context

        Returns:
            Tool execution observation
        """
        try:
            if action == ToolName.WEB_SEARCH:
                from backend.tools.web_search import web_search_tool
                result = await web_search_tool.execute(
                    query=action_input,
                    max_results=5
                )
                return result.output

            elif action == ToolName.RAG_RETRIEVAL:
                from backend.tools.rag_retriever import rag_retriever_tool
                result = await rag_retriever_tool.execute(
                    query=action_input,
                    top_k=5
                )
                return result.output

            elif action == ToolName.PYTHON_CODER:
                from backend.tools.python_coder import python_coder_tool

                # Build react context for code generation
                react_context = self._build_react_context(steps) if steps else None

                result = await python_coder_tool.execute_code_task(
                    query=action_input,
                    file_paths=file_paths,
                    session_id=session_id,
                    react_context=react_context,
                    react_step=len(steps) if steps else None
                )

                if result['success']:
                    return self._format_python_result(result)
                else:
                    return f"Code execution failed: {result.get('error', 'Unknown error')}"

            elif action == ToolName.FILE_ANALYZER:
                from backend.tools.file_analyzer import file_analyzer
                result = await file_analyzer.execute(
                    query=action_input,
                    file_paths=file_paths,
                    user_query=action_input
                )
                return result.output

            elif action == ToolName.VISION_ANALYZER:  # NEW
                from backend.tools.vision_analyzer import vision_analyzer

                # Auto-detect focus based on query keywords
                focus = self._detect_vision_focus(action_input)

                result = await vision_analyzer.execute(
                    query=action_input,
                    file_paths=file_paths,
                    focus=focus
                )
                return result.output

            else:
                return f"Unknown tool: {action}"

        except Exception as e:
            logger.error(f"[ToolExecutor] Error executing {action}: {e}")
            return f"Tool execution error: {str(e)}"

    def _detect_vision_focus(self, query: str) -> str:
        """
        Auto-detect vision analysis focus from query keywords.

        Args:
            query: User's query

        Returns:
            Focus type (general, ocr, objects, scene, chart)
        """
        query_lower = query.lower()

        # OCR focus keywords
        if any(kw in query_lower for kw in ['text', 'read', 'extract', 'ocr', 'written']):
            return "ocr"

        # Chart focus keywords
        if any(kw in query_lower for kw in ['chart', 'graph', 'plot', 'data', 'visualization', 'diagram']):
            return "chart"

        # Object focus keywords
        if any(kw in query_lower for kw in ['object', 'identify', 'detect', 'find', 'locate']):
            return "objects"

        # Scene focus keywords
        if any(kw in query_lower for kw in ['scene', 'setting', 'environment', 'describe', 'what is']):
            return "scene"

        # Default: general
        return "general"

    # ... rest of existing methods ...
```

3. **`backend/config/settings.py:97`** - Add to available_tools list

```python
class Settings(BaseSettings):
    # ... existing settings ...

    # ============================================================================
    # ReAct Agent Configuration
    # ============================================================================

    # Available tools (UPDATED with vision_analyzer)
    available_tools: list[str] = [
        'web_search',
        'rag',
        'python_coder',
        'file_analyzer',  # Analyzes data files + images (metadata)
        'vision_analyzer'  # Deep vision analysis for images
    ]
```

4. **Update tool selection prompts** - `backend/config/prompts/react_agent.py`

```python
def get_thought_action_prompt(
    user_query: str,
    context: str,
    file_paths: Optional[List[str]] = None
) -> str:
    """
    Generate thought-action prompt for ReAct agent.

    UPDATED with vision_analyzer tool description.
    """

    # Detect if images are attached
    has_images = False
    if file_paths:
        from backend.utils.multimodal_message import MultimodalMessageBuilder
        file_types = MultimodalMessageBuilder.separate_files_by_type(file_paths)
        has_images = len(file_types['images']) > 0

    tools_desc = """
Available Tools:

1. **web_search** - Search the web for current information
   - Use when: Need real-time data, news, or online information
   - Input: Search query (e.g., "latest Python 3.12 features")

2. **rag_retrieval** - Search uploaded documents/knowledge base
   - Use when: Need to find information from user's documents
   - Input: Query about document content

3. **python_coder** - Generate and execute Python code
   - Use when: Need to process data, perform calculations, create visualizations
   - Input: Detailed task description
   - Can access uploaded files (CSV, Excel, JSON, images)

4. **file_analyzer** - Analyze uploaded files (CSV, Excel, JSON, images, PDFs)
   - Use when: Need to understand file structure and contents
   - Input: What to analyze (e.g., "summarize this dataset")
   - For images: Returns basic metadata (size, format)
"""

    # Add vision_analyzer if images detected
    if has_images:
        tools_desc += """
5. **vision_analyzer** - Deep visual analysis of images (OCR, scene, objects, charts)
   - Use when: Need to understand IMAGE CONTENT (not just metadata)
   - Input: Question about the image (e.g., "what does this chart show?")
   - Capabilities: OCR text extraction, scene description, chart analysis
   - IMPORTANT: Use this for visual understanding, use file_analyzer for basic metadata
"""

    tools_desc += """
6. **finish** - Complete the task and provide final answer
   - Use when: You have enough information to answer the user's question
   - Input: Not used (will generate final answer from observations)
"""

    prompt = f"""You are a ReAct reasoning agent. Follow this process:

1. **Thought**: Analyze the current situation and decide next action
2. **Action**: Select ONE tool from the available tools
3. **Action Input**: Provide the input for the selected tool

{tools_desc}

**User Query:** {user_query}

{context}

Now, think step by step and select the most appropriate action.

Output EXACTLY in this format:
Thought: [Your reasoning about what to do next]
Action: [tool_name from the list above]
Action Input: [specific input for the tool]
"""

    return prompt
```

**Rationale:**
- Vision tool integrated seamlessly into ReAct's tool selection
- Automatic focus detection from query keywords
- Clear tool descriptions guide agent's decision-making
- Vision tool only shown when images are present (reduces confusion)

---

### **Phase 5: API & Frontend Updates**

#### **5.1 Update Chat API to Handle Multimodal Inputs**

**Files to modify:**
- `backend/api/routes/chat.py:238-396`

**Enhancement:**

```python
@openai_router.post("/chat/completions")
async def chat_completions(
    model: str = Form(...),
    messages: str = Form(...),
    session_id: Optional[str] = Form(None),
    agent_type: str = Form("auto"),
    files: Optional[List[UploadFile]] = File(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    OpenAI-compatible chat completions endpoint with multipart/form-data support.

    ENHANCED: Automatic multimodal detection and vision workflow activation.

    When images are uploaded:
    1. Automatically detects image files
    2. Routes to agentic workflow (minimum: react)
    3. Vision tools become available to the agent

    Parameters:
        - model: Model name (form field)
        - messages: JSON string of message array (form field)
        - session_id: Optional session ID (form field)
        - agent_type: "auto", "react", "plan_execute", or "chat" (form field)
        - files: Optional file uploads (multipart files)
    """
    user_id = current_user["username"]

    # Parse messages JSON
    try:
        messages_list = json.loads(messages)
        parsed_messages = [ChatMessage(**msg) for msg in messages_list]
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid messages JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid message format: {str(e)}")

    # Create new session if none provided
    if not session_id:
        session_id = conversation_store.create_session(user_id)

    # ====== PHASE 1: FILE HANDLING ======
    file_paths, new_files_uploaded = await _handle_file_uploads(files, session_id, user_id)

    # NEW: Detect multimodal input
    from backend.utils.multimodal_message import MultimodalMessageBuilder
    file_types = MultimodalMessageBuilder.separate_files_by_type(file_paths or [])
    has_images = len(file_types['images']) > 0
    has_data_files = len(file_types['data_files']) > 0

    if has_images:
        logger.info(f"[Multimodal] Detected {len(file_types['images'])} image(s) - enabling vision workflow")
    if has_data_files:
        logger.info(f"[Multimodal] Detected {len(file_types['data_files'])} data file(s)")

    # ====== PHASE 2: CLASSIFICATION ======
    user_message = parsed_messages[-1].content if parsed_messages else ""

    # Determine agent type with multimodal awareness
    if agent_type == "auto":
        # Use LLM to classify
        classified_agent_type = await determine_agent_type(
            user_message,
            has_files=bool(file_paths),
            user_id=user_id
        )

        # NEW: Force agentic workflow if images present
        if has_images and classified_agent_type == "chat":
            logger.info("[Multimodal] Images detected - upgrading 'chat' to 'react' for vision analysis")
            classified_agent_type = "react"

    else:
        # Use explicitly specified agent type
        classified_agent_type = agent_type.lower()
        logger.info(f"[Agent Selection] Using explicitly specified agent type: {classified_agent_type}")

    # Validate agent type
    if classified_agent_type not in ["chat", "react", "plan_execute"]:
        logger.warning(f"[Agent Selection] Invalid agent type '{classified_agent_type}', defaulting to 'react'")
        classified_agent_type = "react"

    # ====== PHASE 3: EXECUTION ======
    try:
        agent_metadata = None

        if classified_agent_type == "chat":
            # Use simple chat (with multimodal support if images present)
            logger.info(f"[Agent Execution] Using chat for query: {user_message[:]}")
            response_text = await chat_task.execute(
                messages=parsed_messages,
                session_id=session_id,
                use_memory=(session_id is not None),
                user_id=user_id,
                file_paths=file_paths  # NEW: Pass file paths for multimodal support
            )

        elif classified_agent_type == "react":
            # Use ReAct agent (now with vision tool support)
            logger.info(f"[Agent Execution] Using ReAct agent for query: {user_message[:]}")
            response_text, agent_metadata = await smart_agent_task.execute(
                messages=parsed_messages,
                session_id=session_id,
                user_id=user_id,
                agent_type=AgentType.REACT,
                file_paths=file_paths
            )

        elif classified_agent_type == "plan_execute":
            # Use Plan-and-Execute agent
            logger.info(f"[Agent Execution] Using Plan-and-Execute agent for query: {user_message[:]}")
            response_text, agent_metadata = await smart_agent_task.execute(
                messages=parsed_messages,
                session_id=session_id,
                user_id=user_id,
                agent_type=AgentType.PLAN_EXECUTE,
                file_paths=file_paths
            )
        else:
            raise ValueError(f"Unknown agent type: {classified_agent_type}")

        # ====== PHASE 4: STORAGE ======
        # Save to conversation history with multimodal metadata
        _save_conversation(
            session_id,
            user_message,
            response_text,
            file_paths,
            classified_agent_type,
            agent_metadata
        )

        # Build OpenAI-compatible response
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=model,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            },
            x_session_id=session_id,
            x_agent_metadata=agent_metadata
        )

    except Exception as e:
        # ... existing error handling ...

    finally:
        # ====== CLEANUP ======
        if new_files_uploaded:
            _cleanup_files(file_paths)
            logger.info(f"[Chat] Cleaned up {len(file_paths)} temp files from uploads folder")
```

**Rationale:**
- Automatic vision workflow activation when images uploaded
- Preserves existing text-only behavior
- Multimodal metadata tracked in conversation history
- Seamless integration with existing agent routing

---

#### **5.2 Add Multimodal Examples**

**Update file:** `API_examples.ipynb`

**New Notebook Section:**

```python
# ============================================================
# Example 6: Multimodal Image Analysis with Vision
# ============================================================

print("\n" + "="*60)
print("Example 6: Multimodal - Analyze Charts with Vision")
print("="*60)

# 1. Upload a chart image
print("\n1. Uploading chart image...")
with open("data/sales_chart.png", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/api/files/upload",
        headers={"Authorization": f"Bearer {token}"},
        files={"files": ("sales_chart.png", f, "image/png")}
    )

uploaded_files = response.json()["files"]
image_path = uploaded_files[0]["file_path"]
print(f"✓ Uploaded: {image_path}")

# 2. Ask question about the image (vision analysis)
print("\n2. Asking vision-based question about chart...")
response = requests.post(
    f"{BASE_URL}/v1/chat/completions",
    headers={"Authorization": f"Bearer {token}"},
    data={
        "model": "llama3.2-vision:11b",
        "messages": json.dumps([
            {
                "role": "user",
                "content": "What does this sales chart show? Extract the title, time period, and key insights."
            }
        ]),
        "agent_type": "react"  # Use ReAct for multi-step reasoning
    },
    files=[("files", open(image_path, "rb"))]
)

result = response.json()
answer = result["choices"][0]["message"]["content"]
metadata = result.get("x_agent_metadata", {})

print("\n" + "="*60)
print("VISION ANALYSIS RESULT:")
print("="*60)
print(answer)

if metadata:
    print("\n" + "-"*60)
    print("Agent Execution Trace:")
    print("-"*60)
    for step in metadata.get("steps", []):
        print(f"\nStep {step.get('iteration')}:")
        print(f"  Thought: {step.get('thought', '')[:100]}...")
        print(f"  Action: {step.get('action')}")
        print(f"  Observation: {step.get('observation', '')[:100]}...")


# ============================================================
# Example 7: Extract Text from Screenshot (OCR)
# ============================================================

print("\n\n" + "="*60)
print("Example 7: Multimodal - OCR Text Extraction")
print("="*60)

# Upload screenshot
print("\n1. Uploading screenshot...")
with open("data/screenshot.png", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/api/files/upload",
        headers={"Authorization": f"Bearer {token}"},
        files={"files": ("screenshot.png", f, "image/png")}
    )

image_path = response.json()["files"][0]["file_path"]

# Extract text via vision
print("\n2. Extracting text with OCR...")
response = requests.post(
    f"{BASE_URL}/v1/chat/completions",
    headers={"Authorization": f"Bearer {token}"},
    data={
        "model": "llama3.2-vision:11b",
        "messages": json.dumps([
            {
                "role": "user",
                "content": "Extract all visible text from this screenshot. Preserve formatting."
            }
        ]),
        "agent_type": "react"
    },
    files=[("files", open(image_path, "rb"))]
)

extracted_text = response.json()["choices"][0]["message"]["content"]

print("\n" + "="*60)
print("EXTRACTED TEXT:")
print("="*60)
print(extracted_text)


# ============================================================
# Example 8: Multimodal + Code Generation
# ============================================================

print("\n\n" + "="*60)
print("Example 8: Multimodal - Generate Code to Replicate Chart")
print("="*60)

# Upload chart image
print("\n1. Uploading chart to replicate...")
with open("data/bar_chart.png", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/api/files/upload",
        headers={"Authorization": f"Bearer {token}"},
        files={"files": ("bar_chart.png", f, "image/png")}
    )

image_path = response.json()["files"][0]["file_path"]

# Ask agent to generate code to replicate the chart
print("\n2. Asking agent to generate code to replicate chart...")
response = requests.post(
    f"{BASE_URL}/v1/chat/completions",
    headers={"Authorization": f"Bearer {token}"},
    data={
        "model": "llama3.2-vision:11b",
        "messages": json.dumps([
            {
                "role": "user",
                "content": """Look at this chart and generate Python code to create a similar visualization.

Extract:
- Chart type
- Data values
- Colors
- Labels

Then write Python code (matplotlib) to recreate it."""
            }
        ]),
        "agent_type": "react"
    },
    files=[("files", open(image_path, "rb"))]
)

result = response.json()
answer = result["choices"][0]["message"]["content"]

print("\n" + "="*60)
print("GENERATED CODE:")
print("="*60)
print(answer)


# ============================================================
# Example 9: Multi-Image Comparison
# ============================================================

print("\n\n" + "="*60)
print("Example 9: Multimodal - Compare Multiple Images")
print("="*60)

# Upload multiple images
print("\n1. Uploading images to compare...")
images_to_upload = ["before.png", "after.png"]
uploaded_paths = []

for img_name in images_to_upload:
    with open(f"data/{img_name}", "rb") as f:
        response = requests.post(
            f"{BASE_URL}/api/files/upload",
            headers={"Authorization": f"Bearer {token}"},
            files={"files": (img_name, f, "image/png")}
        )
        uploaded_paths.append(response.json()["files"][0]["file_path"])

print(f"✓ Uploaded {len(uploaded_paths)} images")

# Compare images
print("\n2. Asking agent to compare images...")
response = requests.post(
    f"{BASE_URL}/v1/chat/completions",
    headers={"Authorization": f"Bearer {token}"},
    data={
        "model": "llama3.2-vision:11b",
        "messages": json.dumps([
            {
                "role": "user",
                "content": "Compare these before and after images. What changed?"
            }
        ]),
        "agent_type": "react"
    },
    files=[("files", open(path, "rb")) for path in uploaded_paths]
)

comparison = response.json()["choices"][0]["message"]["content"]

print("\n" + "="*60)
print("COMPARISON RESULT:")
print("="*60)
print(comparison)
```

**Rationale:**
- Demonstrates key multimodal use cases
- Shows integration with existing agent workflows
- Provides copy-paste examples for users
- Covers vision analysis, OCR, code generation, multi-image tasks

---

## 🎯 Implementation Roadmap

### **Week 1: Foundation** (5 days)
**Goal:** Set up infrastructure for multimodal support

- [ ] **Day 1-2:** Model Configuration
  - Update `settings.py` with vision model settings
  - Pull Ollama vision models (`llama3.2-vision:11b`)
  - Test basic Ollama vision API

- [ ] **Day 3-4:** LLM Factory Extension
  - Implement `create_vision_llm()` method
  - Add `supports_vision()` and `get_vision_models()` utilities
  - Write unit tests for vision LLM creation

- [ ] **Day 5:** Multimodal Message Builder
  - Create `backend/utils/multimodal_message.py`
  - Implement base64 encoding and MIME detection
  - Test with sample images

**Deliverables:**
- ✓ Vision models configured and accessible
- ✓ LLMFactory supports vision model creation
- ✓ Image encoding utilities ready

---

### **Week 2: Core Integration** (5 days)
**Goal:** Enable basic multimodal chat capabilities

- [ ] **Day 1-2:** ChatTask Multimodal Support
  - Update `chat_task.py` with `file_paths` parameter
  - Implement `_build_multimodal_conversation()`
  - Test text + image chat interactions

- [ ] **Day 3-4:** File Analyzer Enhancement
  - Add vision-based image analysis to FileAnalyzer
  - Implement `_analyze_image_with_vision()` method
  - Test with charts, screenshots, photos

- [ ] **Day 5:** Integration Testing
  - End-to-end test: upload image → chat with vision
  - Test file type separation logic
  - Performance testing (image encoding overhead)

**Deliverables:**
- ✓ Simple chat works with images
- ✓ FileAnalyzer provides deep visual insights
- ✓ Basic multimodal workflow functional

---

### **Week 3: Agent Integration** (5 days)
**Goal:** Enable ReAct agent to use vision tools

- [ ] **Day 1-2:** Vision Analyzer Tool
  - Create `backend/tools/vision_analyzer/tool.py`
  - Implement focus modes (ocr, chart, scene, objects)
  - Write unit tests

- [ ] **Day 3:** ReAct Tool Registration
  - Add `VISION_ANALYZER` to ToolName enum
  - Update ToolExecutor routing
  - Add to `available_tools` in settings

- [ ] **Day 4:** Prompt Updates
  - Update ReAct thought-action prompts
  - Add vision tool descriptions
  - Implement auto-focus detection

- [ ] **Day 5:** Integration Testing
  - Test ReAct with vision_analyzer tool
  - Test multi-step workflows (vision → code → answer)
  - Debug tool selection logic

**Deliverables:**
- ✓ ReAct agent can use vision_analyzer
- ✓ Tool selection works correctly
- ✓ Multi-step vision workflows functional

---

### **Week 4: Python Coder Enhancement** (5 days)
**Goal:** Enable code generation with image context

- [ ] **Day 1-2:** File Preparation Updates
  - Update `_prepare_files()` for image awareness
  - Add visual context to file metadata
  - Test file type separation

- [ ] **Day 3-4:** Code Generation Context
  - Update code generation prompts with image context
  - Test: "Generate code to display this image"
  - Test: "Extract data from this chart image"

- [ ] **Day 5:** Integration Testing
  - Test image + data file workflows
  - Test code generation referencing images
  - Performance optimization

**Deliverables:**
- ✓ Python coder handles images correctly
- ✓ Code references images by filename
- ✓ Mixed image + data workflows work

---

### **Week 5: API, Testing & Documentation** (5 days)
**Goal:** Finalize multimodal system and document

- [ ] **Day 1:** API Updates
  - Update chat completions endpoint
  - Add automatic multimodal detection
  - Test file upload workflows

- [ ] **Day 2-3:** Examples & Documentation
  - Add multimodal examples to `API_examples.ipynb`
  - Update `CLAUDE.md` with multimodal section
  - Create troubleshooting guide

- [ ] **Day 4:** End-to-End Testing
  - Test all multimodal use cases
  - Performance benchmarking
  - Error handling verification

- [ ] **Day 5:** Deployment Preparation
  - Create migration guide
  - Update requirements.txt if needed
  - Write deployment checklist

**Deliverables:**
- ✓ Complete multimodal system functional
- ✓ Comprehensive documentation
- ✓ Ready for production deployment

---

## 🔧 Key Technical Decisions

### **1. Automatic Vision Detection**
**Decision:** System automatically switches to vision LLM when images are detected in file uploads.

**Rationale:**
- Seamless user experience (no manual model selection)
- Backward compatible (text-only workflows unchanged)
- Efficient (only uses vision LLM when needed)

**Implementation:**
```python
file_types = MultimodalMessageBuilder.separate_files_by_type(file_paths)
has_images = len(file_types['images']) > 0

if has_images:
    llm = LLMFactory.create_vision_llm(user_id=user_id)
else:
    llm = LLMFactory.create_llm(user_id=user_id)
```

---

### **2. Backward Compatibility**
**Decision:** All multimodal features are opt-in via file uploads. Text-only workflows remain unchanged.

**Rationale:**
- No breaking changes to existing API
- Users without vision models can continue using system
- Gradual adoption path

**Example:**
```python
# Text-only (existing behavior)
POST /v1/chat/completions
{
  "messages": [{"role": "user", "content": "Hello"}]
}

# Multimodal (new capability)
POST /v1/chat/completions
{
  "messages": [{"role": "user", "content": "What's in this image?"}],
  "files": [<image file>]
}
```

---

### **3. Separation of Images and Data Files**
**Decision:** Images are handled separately from data files (CSV, Excel, JSON) in the processing pipeline.

**Rationale:**
- Different processing requirements
- Enables specialized routing (images → vision_analyzer, data → python_coder)
- Clear separation of concerns

**Implementation:**
```python
file_types = {
    'images': ['chart.png', 'photo.jpg'],
    'data_files': ['sales.csv', 'data.xlsx']
}

# Route images to vision analysis
for img in file_types['images']:
    vision_result = await vision_analyzer.execute(query, file_paths=[img])

# Route data files to python coder
for data in file_types['data_files']:
    code_result = await python_coder.execute(query, file_paths=[data])
```

---

### **4. Vision Tool Strategy**
**Decision:** Create dedicated `vision_analyzer` tool for ReAct, rather than forcing all tools to be multimodal.

**Rationale:**
- Clear separation: `file_analyzer` for metadata, `vision_analyzer` for visual understanding
- Easier for agent to select correct tool
- Modular design allows independent optimization

**Tool Responsibilities:**
- **file_analyzer:** File metadata (size, format, columns, etc.) + basic image metadata
- **vision_analyzer:** Deep visual understanding (OCR, scene, objects, charts)
- **python_coder:** Generate code to process images/data

---

### **5. File Context Enhancement**
**Decision:** Visual descriptions are added to file metadata as text, not replacing existing text-based analysis.

**Rationale:**
- LLMs can still reason about images via text descriptions
- Works with both vision and non-vision models
- Provides context for code generation

**Example Metadata:**
```python
{
    'file': 'chart.png',
    'type': 'image',
    'width': 1920,
    'height': 1080,
    'format': 'PNG',
    'visual_description': 'Bar chart showing quarterly sales from 2020-2024. Title: "Annual Sales Performance". Red bars represent Q1, blue bars Q2. Y-axis shows revenue in millions.',
    'text_found': ['Annual Sales Performance', 'Q1', 'Q2', 'Q3', 'Q4', '$1.2M'],
    'chart_type': 'bar_chart'
}
```

---

### **6. Vision Focus Auto-Detection**
**Decision:** Automatically detect vision analysis focus (OCR, chart, scene, objects) from query keywords.

**Rationale:**
- Reduces cognitive load on agent
- Optimizes vision model prompts
- Better results with specialized prompts

**Implementation:**
```python
def _detect_vision_focus(query: str) -> str:
    query_lower = query.lower()

    if 'text' in query_lower or 'read' in query_lower:
        return "ocr"
    elif 'chart' in query_lower or 'graph' in query_lower:
        return "chart"
    elif 'object' in query_lower or 'identify' in query_lower:
        return "objects"
    else:
        return "general"
```

---

### **7. Image Attachment Strategy**
**Decision:** Images are attached ONLY to the last user message in a conversation.

**Rationale:**
- Reduces token usage (no redundant image encoding)
- Most vision models expect images in current turn
- Conversation history uses text descriptions of images

**Implementation:**
```python
for i, msg in enumerate(messages):
    is_last_user_message = (i == len(messages) - 1 and msg.role == "user")

    if is_last_user_message and images:
        # Attach images to last user message only
        message = MultimodalMessageBuilder.build_multimodal_message(
            text=msg.content,
            image_paths=images
        )
    else:
        # Text-only message
        message = HumanMessage(content=msg.content)
```

---

## 📊 Expected Benefits

### **1. Image Understanding**
**Capabilities:**
- OCR text extraction from screenshots, documents, signs
- Scene description and context understanding
- Object detection and identification

**Use Cases:**
- Extract data from scanned reports
- Analyze user interface screenshots
- Understand photo content

**Example:**
```
User: "What's the error message in this screenshot?"
Agent: [vision_analyzer] → "The error reads: 'Connection timeout after 30s. Check your network settings.'"
```

---

### **2. Enhanced Code Generation**
**Capabilities:**
- Generate code referencing visual content
- Understand image-based requirements
- Create visualizations matching reference images

**Use Cases:**
- "Generate code to analyze the red bars in chart.png"
- "Recreate this chart with updated data"
- "Display this image with annotations"

**Example:**
```
User: "Generate code to replicate this chart with my data"
Agent:
  [vision_analyzer] → "Bar chart, 5 categories, blue colors"
  [python_coder] → Generates matplotlib code matching visual style
```

---

### **3. Document Processing**
**Capabilities:**
- Extract tables from scanned PDFs
- OCR handwritten notes
- Analyze document layouts

**Use Cases:**
- Process scanned invoices
- Extract data from form images
- Digitize handwritten documents

**Example:**
```
User: "Extract the invoice total from this scanned PDF"
Agent: [vision_analyzer with OCR] → "Invoice Total: $1,234.56"
```

---

### **4. Chart and Data Visualization Analysis**
**Capabilities:**
- Extract data from chart images
- Understand visualization types
- Compare multiple charts

**Use Cases:**
- Analyze competitor charts from screenshots
- Extract data when source files unavailable
- Compare visual trends across images

**Example:**
```
User: "What does this sales chart show?"
Agent: [vision_analyzer with chart focus] →
  "Bar chart showing quarterly sales 2020-2024.
   Q4 2024: $1.2M (highest)
   Overall trend: 15% YoY growth
   Notable: Q2 2023 dip (-8%)"
```

---

### **5. Multimodal RAG**
**Capabilities:**
- Combine text and visual document understanding
- Search across both text and image content
- Provide context-aware answers using both modalities

**Use Cases:**
- Technical documentation with diagrams
- Product catalogs with images
- Research papers with charts

**Example:**
```
User: "How do I assemble this product?" [includes manual image]
Agent:
  [vision_analyzer] → Reads diagram steps
  [rag_retrieval] → Finds text instructions
  → Combines both: "Step 1 shown in diagram: Connect piece A to B..."
```

---

### **6. Visual Question Answering**
**Capabilities:**
- Answer specific questions about image content
- Compare visual elements
- Identify patterns and anomalies

**Use Cases:**
- "Is anyone wearing a helmet in this photo?"
- "Which chart shows higher values?"
- "Are there any errors in this UI?"

**Example:**
```
User: "Does this dashboard show any errors?"
Agent: [vision_analyzer with objects focus] →
  "Yes, 3 error indicators visible:
   - Top right: Red error icon next to 'Server Status'
   - Center: Warning triangle on main chart
   - Bottom: 'Connection Failed' message in footer"
```

---

## ⚠️ Considerations and Limitations

### **1. Model Availability**
**Consideration:** Vision models must be available in Ollama

**Mitigation:**
- Graceful fallback to metadata-only analysis
- Clear error messages if vision model unavailable
- Model capability checking before execution

---

### **2. Token Usage**
**Consideration:** Base64-encoded images increase token usage significantly

**Mitigation:**
- Image compression before encoding
- Attach images only to current turn (not history)
- Configurable image size limits

---

### **3. Performance**
**Consideration:** Vision model inference is slower than text-only

**Mitigation:**
- Async execution for non-blocking performance
- Caching vision analysis results
- Optional vision mode (can be disabled)

---

### **4. Privacy**
**Consideration:** Images may contain sensitive information

**Mitigation:**
- User-specific file storage
- Automatic cleanup after session
- No image data in logs

---

### **5. Vision Model Limitations**
**Consideration:** Vision models may hallucinate or misinterpret images

**Mitigation:**
- Clear confidence indicators
- Verification step in workflows
- Human-in-the-loop for critical decisions

---

## 🧪 Testing Strategy

### **Unit Tests**
```python
# test_multimodal_message.py
def test_encode_image():
    encoded = MultimodalMessageBuilder.encode_image("test.png")
    assert encoded["type"] == "image_url"
    assert "base64" in encoded["image_url"]["url"]

def test_separate_files():
    files = ["chart.png", "data.csv"]
    separated = MultimodalMessageBuilder.separate_files_by_type(files)
    assert separated["images"] == ["chart.png"]
    assert separated["data_files"] == ["data.csv"]

# test_llm_factory.py
def test_create_vision_llm():
    llm = LLMFactory.create_vision_llm()
    assert llm.model == settings.ollama_vision_model

def test_supports_vision():
    assert LLMFactory.supports_vision("llama3.2-vision:11b") == True
    assert LLMFactory.supports_vision("qwen3-coder:30b") == False
```

---

### **Integration Tests**
```python
# test_vision_analyzer.py
async def test_vision_analyzer_chart():
    result = await vision_analyzer.execute(
        query="What data is in this chart?",
        file_paths=["test_chart.png"],
        focus="chart"
    )
    assert result.success == True
    assert "data" in result.output.lower()

# test_chat_task.py
async def test_chat_multimodal():
    messages = [ChatMessage(role="user", content="Describe this image")]
    result = await chat_task.execute(
        messages=messages,
        file_paths=["test.png"],
        user_id="test_user"
    )
    assert len(result) > 0
```

---

### **End-to-End Tests**
```python
# test_multimodal_workflow.py
async def test_image_analysis_workflow():
    # Upload image
    # Ask question via API
    # Verify vision tool was used
    # Check response quality

async def test_code_generation_with_image():
    # Upload chart image
    # Request code to replicate chart
    # Verify code references image
    # Execute code and check output
```

---

## 📚 Documentation Updates

### **Update CLAUDE.md**
Add new section after "Working with Python Code Tool":

```markdown
## Working with Multimodal Inputs (Images)

### Overview
The system supports vision-enabled models for analyzing images alongside text and data files.

### Supported Models
- llama3.2-vision:11b (recommended)
- gemma-2-vision:9b

### Image Capabilities
1. **OCR Text Extraction:** Extract visible text from images
2. **Scene Understanding:** Describe what's in the image
3. **Chart Analysis:** Extract data from graphs and charts
4. **Object Detection:** Identify and locate objects

### Using Vision in API
```python
# Upload image
files = {"files": ("chart.png", open("chart.png", "rb"), "image/png")}
upload_response = requests.post(f"{API_URL}/api/files/upload", files=files)

# Ask question about image
response = requests.post(
    f"{API_URL}/v1/chat/completions",
    data={
        "messages": json.dumps([
            {"role": "user", "content": "What does this chart show?"}
        ]),
        "agent_type": "react"
    },
    files=[("files", open(image_path, "rb"))]
)
```

### Vision Tool in ReAct
The `vision_analyzer` tool provides deep image understanding:
- Automatically activated when images detected
- Multiple focus modes: general, ocr, chart, scene, objects
- Integrated with other tools for multi-step reasoning

### Configuration
```python
# backend/config/settings.py
ollama_vision_model = 'llama3.2-vision:11b'
ollama_vision_enabled = True
```
```

---

## 🚀 Deployment Checklist

- [ ] Pull vision models in production Ollama instance
- [ ] Update `settings.py` with production vision model
- [ ] Test image upload size limits
- [ ] Monitor vision LLM performance metrics
- [ ] Set up image cleanup cron job (if needed)
- [ ] Update API documentation
- [ ] Train team on multimodal capabilities
- [ ] Create user guides with examples
- [ ] Set up monitoring for vision model errors
- [ ] Test with real-world images from users

---

## 📈 Success Metrics

### **Technical Metrics**
- Vision tool usage rate in ReAct workflows
- Average vision analysis time
- Vision analysis success rate
- Token usage increase with images

### **User Metrics**
- User adoption of image uploads
- User satisfaction with vision analysis
- Types of images uploaded (charts, screenshots, photos)
- Feature requests related to vision

### **Business Metrics**
- New use cases enabled by vision
- Reduction in manual data entry (OCR)
- Time saved on chart data extraction
- User retention improvement

---

## 🎓 Training and Adoption

### **For Developers**
- Code review of multimodal implementation
- Workshop on vision tool architecture
- Best practices for vision prompt engineering
- Debugging guide for vision issues

### **For Users**
- Tutorial: "Analyzing Charts with Vision"
- Guide: "Extracting Text from Screenshots"
- FAQ: "When to Use Vision vs File Analyzer"
- Video: "Multimodal Workflows Demo"

---

## 📞 Support and Troubleshooting

### **Common Issues**

**Issue:** Vision model not loading
```bash
# Solution: Check Ollama
ollama list
# If model missing: ollama pull llama3.2-vision:11b
```

**Issue:** Image encoding errors
```python
# Check image format
from PIL import Image
img = Image.open("test.png")
print(f"Format: {img.format}, Size: {img.size}")
```

**Issue:** Vision analysis too slow
```python
# Check model size
# Consider using smaller vision model
ollama_vision_model = 'llama3.2-vision:11b'  # Faster
# vs
ollama_vision_model = 'llama3.2-vision:90b'  # More accurate but slower
```

---

## 🔮 Future Enhancements

### **Phase 6: Advanced Multimodal (Future)**
- [ ] Video frame analysis
- [ ] Audio transcription and analysis
- [ ] Multi-page document processing
- [ ] Real-time webcam analysis
- [ ] Image generation integration (DALL-E, Stable Diffusion)

### **Phase 7: Optimization (Future)**
- [ ] Image compression pipeline
- [ ] Vision result caching
- [ ] Batch image processing
- [ ] Progressive image loading
- [ ] Vision model fine-tuning

---

## 📝 Conclusion

This comprehensive plan provides a **systematic, low-risk path** to adopting multimodal capabilities in your LLM_API system. The modular architecture ensures:

- **Backward compatibility:** Existing text workflows unchanged
- **Incremental adoption:** Implement phase by phase
- **Clear separation:** Images vs data files, vision vs metadata
- **Production-ready:** Error handling, logging, testing
- **Extensible:** Easy to add new modalities (audio, video)

The plan respects your existing v2.0.0 modular architecture and follows the same design patterns, ensuring consistency and maintainability.

**Next Steps:**
1. Review this plan with your team
2. Prioritize phases based on user needs
3. Set up development environment with vision models
4. Begin Week 1 implementation
5. Iterate based on feedback

---

**Document Version:** 1.0.0
**Last Updated:** 2025-01-27
**Status:** Ready for Implementation
