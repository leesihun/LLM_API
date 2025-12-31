"""
Helper script to convert llm_backend.py to async
Run this once to convert the entire file
"""
import re
from pathlib import Path

# Read the current file
backend_file = Path("backend/core/llm_backend.py")
content = backend_file.read_text(encoding='utf-8')

# Define all replacements
replacements = [
    # Ollama _stream_request
    (
        r'    def _stream_request\(self, method: str, url: str, \*\*kwargs\):',
        '    async def _stream_request(self, method: str, url: str, **kwargs):'
    ),
    (
        r'                client = httpx\.Client\(verify=ssl_verify\)',
        '                # Async streaming client will be created in the calling function'
    ),
    (
        r'                stream_context = client\.stream\(method, url, \*\*kwargs\)',
        '                return ssl_verify  # Return the verified SSL option'
    ),

    # Ollama is_available
    (
        r'    def is_available\(self\) -> bool:\s+"""Check if Ollama is running"""\s+try:\s+response = self\._make_request\("GET", f"\{self\.host\}/api/tags", timeout=2\.0\)',
        '    async def is_available(self) -> bool:\n        """Check if Ollama is running"""\n        try:\n            response = await self._make_request("GET", f"{self.host}/api/tags", timeout=2.0)'
    ),

    # Ollama preload_model
    (
        r'    def preload_model\(self,',
        '    async def preload_model(self,'
    ),
    (
        r'            response = self\._make_request\(\s+"POST",\s+f"\{self\.host\}/api/chat",',
        '            response = await self._make_request(\n                "POST",\n                f"{self.host}/api/chat",'
    ),

    # Ollama list_models
    (
        r'    def list_models\(self\) -> List\[str\]:\s+"""List available Ollama models"""\s+try:\s+response = self\._make_request\("GET", f"\{self\.host\}/api/tags", timeout=5\.0\)',
        '    async def list_models(self) -> List[str]:\n        """List available Ollama models"""\n        try:\n            response = await self._make_request("GET", f"{self.host}/api/tags", timeout=5.0)'
    ),

    # Ollama chat
    (
        r'    def chat\(self, messages: List\[Dict\[str, str\]\], model: str, temperature: float = 0\.7\) -> str:\s+"""Non-streaming chat completion"""',
        '    async def chat(self, messages: List[Dict[str, str]], model: str, temperature: float = 0.7) -> str:\n        """Non-streaming chat completion"""'
    ),
    (
        r'        response = self\._make_request\(\s+"POST",\s+f"\{self\.host\}/api/chat",\s+json=payload,\s+timeout=config\.STREAM_TIMEOUT\s+\)',
        '        response = await self._make_request(\n            "POST",\n            f"{self.host}/api/chat",\n            json=payload,\n            timeout=config.STREAM_TIMEOUT\n        )'
    ),

    # Ollama chat_stream
    (
        r'    def chat_stream\(self, messages: List\[Dict\[str, str\]\], model: str, temperature: float = 0\.7\) -> Iterator\[str\]:',
        '    async def chat_stream(self, messages: List[Dict[str, str]], model: str, temperature: float = 0.7) -> AsyncIterator[str]:'
    ),
]

# Apply replacements
for pattern, replacement in replacements:
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)

# Write back
backend_file.write_text(content, encoding='utf-8')
print("✓ Conversion complete")
