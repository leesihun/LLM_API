"""
Notepad Management API Routes

Provides REST API endpoints for managing session notepads, including:
- CRUD operations on entries
- Search and filtering
- Analytics and insights
- Export functionality
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import json

from backend.tools.notepad import SessionNotepad
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)

router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================

class EntryCreate(BaseModel):
    """Model for creating a new notepad entry"""
    task: str
    description: str
    code_file: Optional[str] = None
    variables_saved: Optional[List[str]] = None
    key_outputs: Optional[str] = None
    tags: Optional[List[str]] = None
    importance: str = "medium"
    execution_time_ms: Optional[int] = None
    success_score: Optional[float] = None
    tool_used: Optional[str] = None


class EntryUpdate(BaseModel):
    """Model for updating an entry"""
    task: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    importance: Optional[str] = None
    key_outputs: Optional[str] = None


class SearchFilters(BaseModel):
    """Model for search filters"""
    task: Optional[str] = None
    tags: Optional[List[str]] = None
    importance: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    variables: Optional[List[str]] = None
    tool_used: Optional[str] = None


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/notepad/{session_id}")
async def get_notepad(
    session_id: str,
    include_archived: bool = Query(False, description="Include archived entries")
):
    """
    Get all notepad entries for a session.

    Args:
        session_id: Session identifier
        include_archived: Include archived entries

    Returns:
        Complete notepad data
    """
    try:
        notepad = SessionNotepad.load(session_id)

        entries = notepad.data["entries"]
        if not include_archived:
            entries = [e for e in entries if not e.get("is_archived", False)]

        return {
            "success": True,
            "session_id": session_id,
            "total_entries": len(entries),
            "notepad": {
                **notepad.data,
                "entries": entries
            }
        }
    except Exception as e:
        logger.error(f"[API] Failed to get notepad for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/notepad/{session_id}/entry/{entry_id}")
async def get_entry(session_id: str, entry_id: int):
    """
    Get a specific entry by ID.

    Args:
        session_id: Session identifier
        entry_id: Entry ID

    Returns:
        Entry data
    """
    try:
        notepad = SessionNotepad.load(session_id)
        entry = notepad.get_entry(entry_id)

        if not entry:
            raise HTTPException(status_code=404, detail=f"Entry {entry_id} not found")

        return {
            "success": True,
            "entry": entry
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Failed to get entry {entry_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/notepad/{session_id}")
async def create_entry(session_id: str, entry_data: EntryCreate):
    """
    Create a new notepad entry.

    Args:
        session_id: Session identifier
        entry_data: Entry data

    Returns:
        Created entry ID
    """
    try:
        notepad = SessionNotepad.load(session_id)

        entry_id = notepad.add_entry(
            task=entry_data.task,
            description=entry_data.description,
            code_file=entry_data.code_file,
            variables_saved=entry_data.variables_saved,
            key_outputs=entry_data.key_outputs,
            tags=entry_data.tags,
            importance=entry_data.importance,
            execution_time_ms=entry_data.execution_time_ms,
            success_score=entry_data.success_score,
            tool_used=entry_data.tool_used
        )

        notepad.save()

        return {
            "success": True,
            "entry_id": entry_id,
            "message": f"Entry #{entry_id} created successfully"
        }
    except Exception as e:
        logger.error(f"[API] Failed to create entry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/notepad/{session_id}/entry/{entry_id}")
async def update_entry(session_id: str, entry_id: int, updates: EntryUpdate):
    """
    Update an existing entry.

    Args:
        session_id: Session identifier
        entry_id: Entry ID
        updates: Fields to update

    Returns:
        Success status
    """
    try:
        notepad = SessionNotepad.load(session_id)

        # Convert to dict and filter None values
        update_dict = {k: v for k, v in updates.dict().items() if v is not None}

        if not update_dict:
            return {
                "success": True,
                "message": "No updates provided"
            }

        success = notepad.update_entry(entry_id, **update_dict)

        if success:
            notepad.save()
            return {
                "success": True,
                "message": f"Entry #{entry_id} updated successfully"
            }
        else:
            raise HTTPException(status_code=404, detail=f"Entry {entry_id} not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Failed to update entry {entry_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/notepad/{session_id}/entry/{entry_id}")
async def delete_entry(
    session_id: str,
    entry_id: int,
    soft_delete: bool = Query(True, description="Archive instead of permanent delete")
):
    """
    Delete or archive an entry.

    Args:
        session_id: Session identifier
        entry_id: Entry ID
        soft_delete: If True, archive instead of delete

    Returns:
        Success status
    """
    try:
        notepad = SessionNotepad.load(session_id)

        success = notepad.delete_entry(entry_id, soft_delete=soft_delete)

        if success:
            notepad.save()
            action = "archived" if soft_delete else "deleted"
            return {
                "success": True,
                "message": f"Entry #{entry_id} {action} successfully"
            }
        else:
            raise HTTPException(status_code=404, detail=f"Entry {entry_id} not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Failed to delete entry {entry_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/notepad/{session_id}/search")
async def search_entries(
    session_id: str,
    query: Optional[str] = None,
    filters: Optional[SearchFilters] = None,
    include_archived: bool = False
):
    """
    Search notepad entries.

    Args:
        session_id: Session identifier
        query: Text query
        filters: Search filters
        include_archived: Include archived entries

    Returns:
        Matching entries
    """
    try:
        notepad = SessionNotepad.load(session_id)

        filter_dict = filters.dict(exclude_none=True) if filters else {}

        results = notepad.search_entries(
            query=query,
            filters=filter_dict,
            include_archived=include_archived
        )

        return {
            "success": True,
            "count": len(results),
            "entries": results
        }
    except Exception as e:
        logger.error(f"[API] Failed to search entries: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/notepad/{session_id}/variables")
async def list_variables(session_id: str):
    """
    List all variables mentioned in notepad entries.

    Args:
        session_id: Session identifier

    Returns:
        Variable usage patterns
    """
    try:
        notepad = SessionNotepad.load(session_id)
        patterns = notepad.get_variable_usage_patterns()

        return {
            "success": True,
            **patterns
        }
    except Exception as e:
        logger.error(f"[API] Failed to list variables: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/notepad/{session_id}/variable/{variable_name}")
async def get_variable_entries(session_id: str, variable_name: str):
    """
    Find all entries that used a specific variable.

    Args:
        session_id: Session identifier
        variable_name: Variable name

    Returns:
        List of entries
    """
    try:
        notepad = SessionNotepad.load(session_id)
        entries = notepad.find_entries_with_variable(variable_name)

        return {
            "success": True,
            "variable": variable_name,
            "count": len(entries),
            "entries": entries
        }
    except Exception as e:
        logger.error(f"[API] Failed to find entries for variable {variable_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/notepad/{session_id}/stats")
async def get_stats(session_id: str):
    """
    Get session statistics and analytics.

    Args:
        session_id: Session identifier

    Returns:
        Session statistics
    """
    try:
        notepad = SessionNotepad.load(session_id)
        stats = notepad.get_session_stats()
        patterns = notepad.get_task_sequence_patterns()

        return {
            "success": True,
            "stats": stats,
            "task_sequences": patterns
        }
    except Exception as e:
        logger.error(f"[API] Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/notepad/{session_id}/export/markdown")
async def export_markdown(
    session_id: str,
    include_code: bool = Query(True, description="Include code in export"),
    include_archived: bool = Query(False, description="Include archived entries")
):
    """
    Export notepad to Markdown format.

    Args:
        session_id: Session identifier
        include_code: Include code in export
        include_archived: Include archived entries

    Returns:
        Markdown content
    """
    try:
        notepad = SessionNotepad.load(session_id)
        markdown = notepad.export_to_markdown(
            include_code=include_code,
            include_archived=include_archived
        )

        return {
            "success": True,
            "format": "markdown",
            "content": markdown
        }
    except Exception as e:
        logger.error(f"[API] Failed to export markdown: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/notepad/{session_id}/export/jupyter")
async def export_jupyter(session_id: str):
    """
    Export notepad to Jupyter notebook format.

    Args:
        session_id: Session identifier

    Returns:
        Jupyter notebook JSON
    """
    try:
        notepad = SessionNotepad.load(session_id)
        notebook = notepad.export_to_jupyter_notebook()

        return {
            "success": True,
            "format": "jupyter",
            "notebook": notebook
        }
    except Exception as e:
        logger.error(f"[API] Failed to export jupyter: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/notepad/{session_id}/export/json")
async def export_json(
    session_id: str,
    include_archived: bool = Query(False, description="Include archived entries")
):
    """
    Export notepad to JSON format.

    Args:
        session_id: Session identifier
        include_archived: Include archived entries

    Returns:
        Complete notepad JSON
    """
    try:
        notepad = SessionNotepad.load(session_id)
        data = notepad.export_to_json(include_archived=include_archived)

        return {
            "success": True,
            "format": "json",
            "data": data
        }
    except Exception as e:
        logger.error(f"[API] Failed to export json: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/notepad/{session_id}/code/{code_file}")
async def get_code_content(session_id: str, code_file: str):
    """
    Get code content from a saved file.

    Args:
        session_id: Session identifier
        code_file: Code filename

    Returns:
        Code content
    """
    try:
        notepad = SessionNotepad.load(session_id)
        code = notepad.get_code_content(code_file)

        if code is None:
            raise HTTPException(status_code=404, detail=f"Code file {code_file} not found")

        return {
            "success": True,
            "file": code_file,
            "content": code
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Failed to get code content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/notepad/{session_id}/code/compare")
async def compare_code(
    session_id: str,
    file1: str = Query(..., description="First code file"),
    file2: str = Query(..., description="Second code file")
):
    """
    Compare two code versions.

    Args:
        session_id: Session identifier
        file1: First code filename
        file2: Second code filename

    Returns:
        Unified diff
    """
    try:
        notepad = SessionNotepad.load(session_id)
        diff = notepad.compare_code_versions(file1, file2)

        return {
            "success": True,
            "file1": file1,
            "file2": file2,
            "diff": diff
        }
    except Exception as e:
        logger.error(f"[API] Failed to compare code: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/notepad/{session_id}/code/evolution/{task}")
async def get_code_evolution(session_id: str, task: str):
    """
    Get code evolution for a specific task category.

    Args:
        session_id: Session identifier
        task: Task category

    Returns:
        List of code versions
    """
    try:
        notepad = SessionNotepad.load(session_id)
        evolution = notepad.get_code_evolution(task)

        return {
            "success": True,
            "task": task,
            "versions": evolution
        }
    except Exception as e:
        logger.error(f"[API] Failed to get code evolution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/notepad/{session_id}/entry/{entry_id}/history")
async def get_entry_history(session_id: str, entry_id: int):
    """
    Get version history for an entry.

    Args:
        session_id: Session identifier
        entry_id: Entry ID

    Returns:
        Version history
    """
    try:
        notepad = SessionNotepad.load(session_id)
        history = notepad.get_entry_history(entry_id)

        return {
            "success": True,
            "entry_id": entry_id,
            "versions": history
        }
    except Exception as e:
        logger.error(f"[API] Failed to get entry history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
