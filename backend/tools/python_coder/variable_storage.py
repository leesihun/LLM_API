"""
Variable Storage Module

Handles type-specific serialization and deserialization of variables from code execution.
Supports DataFrames (Parquet), numpy arrays (.npy), simple types (JSON), and matplotlib figures (PNG).
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

from backend.config.settings import settings
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)


class VariableStorage:
    """
    Type-specific variable serialization and metadata management.
    
    Supported types:
    - pandas.DataFrame → .parquet
    - numpy.ndarray → .npy
    - dict/list/str/int/float/bool → .json
    - matplotlib.figure.Figure → .png
    - Other objects → metadata only (no persistence)
    """
    
    @staticmethod
    def save_variables(session_id: str, namespace_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and save variables from execution namespace with type-specific serialization.
        
        Args:
            session_id: Session ID for directory path
            namespace_dict: Dictionary of variable name -> value from execution
            
        Returns:
            Dict of variable metadata including load instructions
        """
        if not session_id:
            logger.warning("[VariableStorage] No session_id provided, skipping variable save")
            return {}
        
        # Setup paths
        session_path = Path(settings.python_code_execution_dir) / session_id
        variables_dir = session_path / "variables"
        variables_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_path = variables_dir / "variables_metadata.json"
        
        # Load existing metadata or create new
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.warning(f"[VariableStorage] Failed to load existing metadata: {e}")
                metadata = {}
        else:
            metadata = {}
        
        # Filter and save each variable
        filtered_vars = VariableStorage._extract_user_variables(namespace_dict)
        
        logger.info(f"[VariableStorage] Saving {len(filtered_vars)} variable(s) for session {session_id}")
        
        for var_name, var_value in filtered_vars.items():
            try:
                var_metadata = VariableStorage._serialize_variable(
                    var_name, var_value, variables_dir
                )
                if var_metadata:
                    metadata[var_name] = var_metadata
                    logger.info(f"[VariableStorage] Saved: {var_name} ({var_metadata['type']})")
            except Exception as e:
                logger.error(f"[VariableStorage] Failed to save variable '{var_name}': {e}")
        
        # Save metadata
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"[VariableStorage] Metadata saved to {metadata_path.name}")
        except Exception as e:
            logger.error(f"[VariableStorage] Failed to save metadata: {e}")
        
        return metadata
    
    @staticmethod
    def get_metadata(session_id: str) -> Dict[str, Any]:
        """
        Get catalog of available variables with loading instructions.
        
        Args:
            session_id: Session ID for directory path
            
        Returns:
            Dict of variable metadata
        """
        if not session_id:
            return {}
        
        session_path = Path(settings.python_code_execution_dir) / session_id
        metadata_path = session_path / "variables" / "variables_metadata.json"
        
        if not metadata_path.exists():
            return {}
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"[VariableStorage] Failed to load metadata: {e}")
            return {}
    
    @staticmethod
    def _extract_user_variables(namespace: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract only user-defined data variables from namespace.
        
        Filters out:
        - Built-ins (starting with _)
        - Modules
        - Functions/callables
        - Classes
        
        Args:
            namespace: Execution namespace dictionary
            
        Returns:
            Filtered dictionary of data variables
        """
        import types
        
        # Import pandas and numpy for type checking
        try:
            import pandas as pd
            has_pandas = True
        except ImportError:
            has_pandas = False
        
        try:
            import numpy as np
            has_numpy = True
        except ImportError:
            has_numpy = False
        
        # Define keep types
        keep_types = [dict, list, str, int, float, bool, tuple, set]
        if has_pandas:
            keep_types.append(pd.DataFrame)
            keep_types.append(pd.Series)
        if has_numpy:
            keep_types.append(np.ndarray)
        
        keep_types = tuple(keep_types)
        
        filtered = {}
        for key, value in namespace.items():
            # Skip built-ins and private variables
            if key.startswith('_'):
                continue
            
            # Skip modules
            if isinstance(value, types.ModuleType):
                continue
            
            # Skip functions and classes
            if callable(value) and not isinstance(value, type):
                continue
            
            # Skip types/classes
            if isinstance(value, type):
                continue
            
            # Keep supported data types
            if isinstance(value, keep_types):
                filtered[key] = value
            elif has_matplotlib_figure(value):
                filtered[key] = value
        
        return filtered
    
    @staticmethod
    def _serialize_variable(var_name: str, var_value: Any, variables_dir: Path) -> Optional[Dict[str, Any]]:
        """
        Serialize variable with type-specific handler.
        
        Args:
            var_name: Variable name
            var_value: Variable value
            variables_dir: Directory to save variable files
            
        Returns:
            Metadata dict for the variable, or None if unsupported
        """
        var_type = type(var_value).__name__
        module = type(var_value).__module__
        full_type = f"{module}.{var_type}" if module != "builtins" else var_type
        
        timestamp = datetime.now().isoformat()
        
        # Try pandas DataFrame
        try:
            import pandas as pd
            if isinstance(var_value, pd.DataFrame):
                return VariableStorage._save_dataframe(var_name, var_value, variables_dir, timestamp)
        except ImportError:
            pass
        
        # Try numpy array
        try:
            import numpy as np
            if isinstance(var_value, np.ndarray):
                return VariableStorage._save_numpy_array(var_name, var_value, variables_dir, timestamp)
        except ImportError:
            pass
        
        # Try matplotlib figure
        if has_matplotlib_figure(var_value):
            return VariableStorage._save_matplotlib_figure(var_name, var_value, variables_dir, timestamp)
        
        # Simple types (dict, list, str, int, float, bool)
        if isinstance(var_value, (dict, list, str, int, float, bool, tuple, set)):
            return VariableStorage._save_json_serializable(var_name, var_value, variables_dir, timestamp)
        
        # Unsupported type - save metadata only
        logger.warning(f"[VariableStorage] Unsupported type for '{var_name}': {full_type}")
        return {
            "type": full_type,
            "file": None,
            "saved": False,
            "reason": "Unsupported type for serialization",
            "timestamp": timestamp
        }
    
    @staticmethod
    def _save_dataframe(var_name: str, df, variables_dir: Path, timestamp: str) -> Dict[str, Any]:
        """Save pandas DataFrame as Parquet."""
        filename = f"{var_name}.parquet"
        filepath = variables_dir / filename
        
        df.to_parquet(filepath, index=False)
        
        size_mb = filepath.stat().st_size / (1024 * 1024)
        
        return {
            "type": "pandas.DataFrame",
            "file": filename,
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "size_mb": round(size_mb, 2),
            "load_code": f"{var_name} = pd.read_parquet('variables/{filename}')",
            "timestamp": timestamp
        }
    
    @staticmethod
    def _save_numpy_array(var_name: str, arr, variables_dir: Path, timestamp: str) -> Dict[str, Any]:
        """Save numpy array as .npy."""
        import numpy as np
        
        filename = f"{var_name}.npy"
        filepath = variables_dir / filename
        
        np.save(filepath, arr)
        
        size_mb = filepath.stat().st_size / (1024 * 1024)
        
        return {
            "type": "numpy.ndarray",
            "file": filename,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "size_mb": round(size_mb, 2),
            "load_code": f"{var_name} = np.load('variables/{filename}')",
            "timestamp": timestamp
        }
    
    @staticmethod
    def _save_json_serializable(var_name: str, value: Any, variables_dir: Path, timestamp: str) -> Dict[str, Any]:
        """Save JSON-serializable types (dict, list, str, int, float, bool)."""
        filename = f"{var_name}.json"
        filepath = variables_dir / filename
        
        # Convert tuples/sets to lists for JSON
        if isinstance(value, (tuple, set)):
            value = list(value)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(value, f, indent=2, ensure_ascii=False)
        
        size_mb = filepath.stat().st_size / (1024 * 1024)
        
        metadata = {
            "type": type(value).__name__,
            "file": filename,
            "size_mb": round(size_mb, 4),
            "timestamp": timestamp
        }
        
        # Add type-specific metadata
        if isinstance(value, dict):
            metadata["keys"] = list(value.keys())[:10]  # First 10 keys
            metadata["load_code"] = f"with open('variables/{filename}') as f: {var_name} = json.load(f)"
        elif isinstance(value, list):
            metadata["length"] = len(value)
            metadata["load_code"] = f"with open('variables/{filename}') as f: {var_name} = json.load(f)"
        else:
            metadata["value_preview"] = str(value)[:100]
            metadata["load_code"] = f"with open('variables/{filename}') as f: {var_name} = json.load(f)"
        
        return metadata
    
    @staticmethod
    def _save_matplotlib_figure(var_name: str, fig, variables_dir: Path, timestamp: str) -> Dict[str, Any]:
        """Save matplotlib figure as PNG."""
        filename = f"{var_name}.png"
        filepath = variables_dir / filename
        
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        
        size_mb = filepath.stat().st_size / (1024 * 1024)
        
        return {
            "type": "matplotlib.figure.Figure",
            "file": filename,
            "size_mb": round(size_mb, 2),
            "saved_as": "image",
            "note": "Figure saved as image, cannot be reloaded as Figure object",
            "timestamp": timestamp
        }


def has_matplotlib_figure(obj: Any) -> bool:
    """Check if object is a matplotlib Figure."""
    try:
        import matplotlib.figure
        return isinstance(obj, matplotlib.figure.Figure)
    except ImportError:
        return False

