"""
File handling utilities for uploads
"""
import shutil
from pathlib import Path
from typing import List
from fastapi import UploadFile

import config


def save_uploaded_files(
    files: List[UploadFile],
    username: str,
    session_id: str
) -> List[str]:
    """
    Save uploaded files to both persistent and scratch directories

    Args:
        files: List of uploaded files
        username: Username for persistent storage
        session_id: Session ID for scratch storage

    Returns:
        List of file paths in scratch directory
    """
    scratch_paths = []

    # Create directories
    user_upload_dir = config.UPLOAD_DIR / username
    session_scratch_dir = config.SCRATCH_DIR / session_id
    user_upload_dir.mkdir(parents=True, exist_ok=True)
    session_scratch_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        if file.filename:
            # Save to user's persistent upload directory
            user_file_path = user_upload_dir / file.filename
            with open(user_file_path, 'wb') as f:
                shutil.copyfileobj(file.file, f)

            # Also copy to session scratch directory
            file.file.seek(0)  # Reset file pointer
            scratch_file_path = session_scratch_dir / file.filename
            with open(scratch_file_path, 'wb') as f:
                shutil.copyfileobj(file.file, f)

            scratch_paths.append(str(scratch_file_path))

    return scratch_paths


def read_file_content(file_path: str) -> str:
    """
    Read file content as text (for adding to LLM context)

    Args:
        file_path: Path to the file

    Returns:
        File content as string
    """
    try:
        path = Path(file_path)

        # Handle text files
        if path.suffix in ['.txt', '.md', '.json', '.csv', '.py', '.js', '.html', '.xml']:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()

        # Handle Excel files
        elif path.suffix in ['.xlsx', '.xls']:
            import pandas as pd
            df = pd.read_excel(path)
            return df.to_string()

        # Handle other file types
        else:
            return f"[Binary file: {path.name}]"

    except Exception as e:
        return f"[Error reading file {file_path}: {str(e)}]"


def cleanup_session_files(session_id: str):
    """
    Clean up scratch files for a session

    Args:
        session_id: Session ID to clean up
    """
    session_scratch_dir = config.SCRATCH_DIR / session_id
    if session_scratch_dir.exists():
        shutil.rmtree(session_scratch_dir)
