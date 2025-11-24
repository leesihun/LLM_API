"""
Quick import validation script - tests all modularized components
WITHOUT actually loading heavy dependencies (Ollama, LangChain, etc.)
"""
import sys
import importlib.util
from pathlib import Path

def can_import(module_path: str) -> tuple[bool, str]:
    """Check if a module can be found (without actually importing it)"""
    try:
        spec = importlib.util.find_spec(module_path)
        if spec is None:
            return False, f"Module not found: {module_path}"
        return True, "OK"
    except (ImportError, ModuleNotFoundError, ValueError) as e:
        return False, str(e)

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("Import Validation Test")
print("=" * 70)
print()

modules_to_test = [
    # ReAct Agent
    ("backend.tasks.react", "ReAct package"),
    ("backend.tasks.react.agent", "ReAct agent module"),
    ("backend.tasks.react.models", "ReAct models"),

    # Python Coder
    ("backend.tools.python_coder", "Python Coder package"),
    ("backend.tools.python_coder.orchestrator", "Python Coder orchestrator"),
    ("backend.tools.python_coder.executor.core", "Python Coder executor"),

    # File Analyzer
    ("backend.tools.file_analyzer", "File Analyzer package"),
    ("backend.tools.file_analyzer.analyzer", "File Analyzer main"),

    # Web Search
    ("backend.tools.web_search", "Web Search package"),
    ("backend.tools.web_search.searcher", "Web Search searcher"),

    # RAG Retriever
    ("backend.tools.rag_retriever", "RAG Retriever package"),
    ("backend.tools.rag_retriever.tool", "RAG Retriever tool"),
    ("backend.tools.rag_retriever.retriever", "RAG Retriever core"),

    # API Routes
    ("backend.api.routes", "API Routes package"),
]

all_ok = True
for module_path, description in modules_to_test:
    ok, message = can_import(module_path)
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {description:30s} ({module_path})")
    if not ok:
        print(f"  Error: {message}")
        all_ok = False

print()
print("=" * 70)
if all_ok:
    print("SUCCESS: All modules can be imported!")
else:
    print("ERROR: Some modules have import issues")
print("=" * 70)
