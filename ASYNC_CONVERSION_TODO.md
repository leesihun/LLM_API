# Async Conversion TODO

## Status: Partially Complete

### ✅ Completed
1. File locking added to `ConversationStore` (database.py)
2. Abstract base class `LLMBackend` converted to async
3. `OllamaBackend` core methods converted to async:
   - `_make_request()` ✅
   - `chat()` ✅
   - `is_available()` ✅
   - `list_models()` ✅
   - `preload_model()` ✅

### ⚠️ Partially Complete (Needs Fixing)
4. `OllamaBackend.chat_stream()` - broken, needs proper async generator
5. `OllamaBackend._stream_request()` - broken, just returns SSL option

### ❌ TODO
6. `LlamaCppBackend` - ALL methods still synchronous
   - Need to convert: `_make_request()`, `chat()`, `chat_stream()`, `is_available()`, `list_models()`

7. `AutoLLMBackend` - ALL methods still synchronous
   - Need to convert: `_get_backend()`, `is_available()`, `chat()`, `chat_stream()`, `list_models()`

8. `backend/core/llm_interceptor.py` - Needs async wrapper

9. `backend/agents/base_agent.py`:
   - `call_llm()` → `async def call_llm()`
   - `call_tool()` → `async def call_tool()`
   - `run()` → `async def run()`

10. All agent implementations:
    - `ChatAgent.run()`
    - `ReActAgent.run()` + internal methods
    - `PlanExecuteAgent.run()` + internal methods
    - `AutoAgent.run()`

11. `backend/api/routes/chat.py`:
    - Update to `await agent.run()`

## Current Blocker

The file `backend/core/llm_backend.py` is partially converted and has syntax errors in streaming methods.
Need to fix or temporarily disable streaming to get basic async working.

## Recommendation

**Option A: Fix Everything** (2-3 hours)
- Complete all conversions above
- Fix streaming properly
- Full async system

**Option B: Minimal Viable Async** (30 minutes)
- Skip streaming conversion (keep synchronous for now)
- Only convert non-streaming `.chat()` path
- Agents only use non-streaming, so this unblocks concurrency
- Add TODO comments for streaming

**Option C: Worker Processes** (5 minutes)
- Abandon async conversion
- Add `workers=4` to uvicorn config
- Each worker = separate process
- Good enough for 2-4 concurrent users

## My Recommendation
Start with **Option B** - get basic concurrency working ASAP, then iterate.
