
import sys
import builtins

# Allowed modules
ALLOWED_MODULES = ['math', 'random', 'datetime', 'json', 're', 'collections', 'itertools', 'functools', 'operator', 'string', 'time', 'numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy', 'sklearn', 'plotly', 'PIL', 'requests', 'pathlib', 'os', 'sys']

# Store original import
_original_import = builtins.__import__

def restricted_import(name, *args, **kwargs):
    """Restrict imports to allowed modules"""
    base_module = name.split('.')[0]
    if base_module not in ALLOWED_MODULES:
        raise ImportError(f"Module '{name}' is not allowed in sandboxed execution")
    return _original_import(name, *args, **kwargs)

# Replace built-in import
builtins.__import__ = restricted_import

# User code starts here
try:
    result = 11.951 / 3.751
    Action Input: print(result)
    =========================================================
except Exception as e:
    import traceback
    print("EXECUTION ERROR:", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)
