"""
Backward Compatibility Shim for React.py
=========================================
This file provides backward compatibility for code using the old monolithic React.py

DEPRECATED: Please use 'from backend.tasks.react import ...' instead

The modular implementation is in backend/tasks/react/
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from 'backend.tasks.React' is deprecated. "
    "Use 'from backend.tasks.react import ReActAgent' instead. "
    "This compatibility shim will be removed in v3.0.0",
    DeprecationWarning,
    stacklevel=2
)

# Import from new modular location
from backend.tasks.react import (
    ReActAgent,
    ReActAgentFactory,
    ReActStep,
    ReActResult,
    ToolName,
    react_agent,  # Singleton instance
)

# Export everything for backward compatibility
__all__ = [
    'ReActAgent',
    'ReActAgentFactory',
    'ReActStep',
    'ReActResult',
    'ToolName',
    'react_agent',
]
