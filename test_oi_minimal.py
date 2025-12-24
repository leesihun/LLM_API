"""
Minimal test to reproduce OpenInterpreter + Ollama error
"""
from interpreter import interpreter

print("Setting up interpreter...")
interpreter.llm.model = "ollama/qwen2.5-coder:7b"
interpreter.llm.api_base = "http://localhost:11434"
interpreter.auto_run = True

print(f"Model: {interpreter.llm.model}")
print(f"API Base: {interpreter.llm.api_base}")
print("\nTesting chat...")

try:
    response = interpreter.chat("print('hello world')", display=False, stream=False)
    print("SUCCESS!")
    for chunk in response:
        print(f"Chunk: {chunk}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
