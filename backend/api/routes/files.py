"""
File Routes
Handles file upload, download, and management operations
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import Dict, Any, List
from pathlib import Path
import uuid

from backend.api.dependencies import get_current_user
from backend.config.settings import settings
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ============================================================================
# Router Setup
# ============================================================================

files_router = APIRouter(prefix="/api/files", tags=["File Management"])


# ============================================================================
# File Management Endpoints
# ============================================================================

@files_router.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Upload files for the current user

    Returns:
        - file_paths: List of uploaded file paths
        - file_info: Metadata about uploaded files
    """
    user_id = current_user["username"]
    uploads_path = Path(settings.uploads_path) / user_id
    uploads_path.mkdir(parents=True, exist_ok=True)

    uploaded_files = []

    for file in files:
        try:
            # Generate unique filename with temp_ prefix (consistent with chat uploads)
            file_id = uuid.uuid4().hex[:8]
            filename = f"temp_{file_id}_{file.filename}"
            file_path = uploads_path / filename

            # Save file
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)

            uploaded_files.append({
                "filename": file.filename,
                "file_path": str(file_path),
                "size": len(content),
                "content_type": file.content_type
            })

            logger.info(f"[Files] Uploaded: {filename} ({len(content)} bytes)")

        except Exception as e:
            logger.error(f"[Files] Error uploading {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to upload {file.filename}")

    return {
        "success": True,
        "files": uploaded_files,
        "count": len(uploaded_files)
    }


@files_router.get("/list")
async def list_files(current_user: Dict[str, Any] = Depends(get_current_user)):
    """List all files for the current user"""
    user_id = current_user["username"]
    uploads_path = Path(settings.uploads_path) / user_id

    if not uploads_path.exists():
        return {"files": [], "count": 0}

    files = []
    for file_path in uploads_path.iterdir():
        if file_path.is_file():
            files.append({
                "filename": file_path.name,
                "path": str(file_path),
                "size": file_path.stat().st_size,
                "modified": file_path.stat().st_mtime
            })

    return {
        "files": files,
        "count": len(files)
    }


@files_router.delete("/{filename}")
async def delete_file(filename: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Delete a specific file"""
    user_id = current_user["username"]
    file_path = Path(settings.uploads_path) / user_id / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        file_path.unlink()
        logger.info(f"[Files] Deleted: {filename} for user {user_id}")
        return {"success": True, "filename": filename}
    except Exception as e:
        logger.error(f"[Files] Error deleting {filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete file")
