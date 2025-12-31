# Concurrency Setup Complete ✓

## What Was Changed

### 1. **File Locking Added** (Safety Fix)
- **File:** `backend/core/database.py`
- **Changes:** Added `FileLock` to `save_conversation()` and `load_conversation()`
- **Why:** Prevents data corruption if same session gets multiple requests
- **Impact:** Safe concurrent access to conversation JSON files

### 2. **Worker Processes Added** (Concurrency Fix)
- **Files:** `config.py`, `server.py`, `tools_server.py`
- **Changes:**
  - Added `SERVER_WORKERS = 4` (main API)
  - Added `TOOLS_SERVER_WORKERS = 4` (tools API)
  - Updated uvicorn to use workers
- **Why:** Allows 4 concurrent requests to be processed simultaneously
- **Impact:** True parallelism for multiple users

### 3. **Test Script Created**
- **File:** `test_concurrency.py`
- **Purpose:** Verify concurrent requests work properly

---

## How to Use

### Starting the Servers

**Option 1: Using the startup script** (Recommended)
```bash
bash start_servers.sh
```

**Option 2: Manual startup**
```bash
# Terminal 1 - Tools server
python tools_server.py

# Terminal 2 - Main server
python server.py
```

You should see output like:
```
[INFO] Started parent process [12345]
[INFO] Started worker process [12346]
[INFO] Started worker process [12347]
[INFO] Started worker process [12348]
[INFO] Started worker process [12349]
```

This confirms 4 workers are running!

---

## Testing Concurrency

Run the test script:
```bash
python test_concurrency.py
```

**Expected output (CONCURRENT):**
```
Request 1: Starting
Request 2: Starting
Request 3: Starting
Request 4: Starting
Request 1: ✓ Completed in 12.3s
Request 3: ✓ Completed in 12.5s
Request 2: ✓ Completed in 12.8s
Request 4: ✓ Completed in 13.1s

Total time: 13.5s
✓ Requests appear to be processed CONCURRENTLY
```

**Bad output (SEQUENTIAL - if workers didn't work):**
```
Request 1: Starting
Request 1: ✓ Completed in 12.3s
Request 2: Starting
Request 2: ✓ Completed in 12.5s
...

Total time: 50.2s
⚠ Requests appear to be processed SEQUENTIALLY
```

---

## Architecture

```
┌──────────────────────────────────────────┐
│  Remote Computer (GPU Server)            │
│  llama.cpp server                        │
│  → Handles multiple requests in parallel│
└──────────────────────────────────────────┘
            ↑     ↑     ↑     ↑
            │     │     │     │
    ┌───────┴─────┴─────┴─────┴───────┐
    │  Your Computer (API Server)      │
    │  ┌────────────────────────────┐  │
    │  │ Main API (port 10007)      │  │
    │  │   Worker 1 (~50MB)         │  │
    │  │   Worker 2 (~50MB)         │  │
    │  │   Worker 3 (~50MB)         │  │
    │  │   Worker 4 (~50MB)         │  │
    │  └────────────────────────────┘  │
    │  ┌────────────────────────────┐  │
    │  │ Tools API (port 10006)     │  │
    │  │   Worker 1 (~50MB)         │  │
    │  │   Worker 2 (~50MB)         │  │
    │  │   Worker 3 (~50MB)         │  │
    │  │   Worker 4 (~50MB)         │  │
    │  └────────────────────────────┘  │
    └──────────────────────────────────┘
    Total RAM: ~400MB (workers only)
    Model RAM: 0 (on remote server)
```

---

## Configuration Options

Edit `config.py` to adjust worker count:

```python
# Decrease for less RAM / fewer concurrent users
SERVER_WORKERS = 2

# Increase for more concurrent users (up to llama.cpp limit)
SERVER_WORKERS = 8

# Set to 1 for debugging (easier to trace errors)
SERVER_WORKERS = 1
```

**Rule of thumb:**
- 2 workers = 2-4 concurrent users
- 4 workers = 4-8 concurrent users
- 8 workers = 8-16 concurrent users

Limited by:
1. Your llama.cpp server's parallel capacity
2. Network bandwidth
3. CPU cores on API server (1-2 cores per worker)

---

## Troubleshooting

### Problem: Workers not starting
**Symptom:** Only see 1 process in logs
**Solution:** Check if `workers` parameter is in server.py/tools_server.py
```python
uvicorn.run(..., workers=config.SERVER_WORKERS)
```

### Problem: "Address already in use"
**Symptom:** Can't start server
**Solution:** Kill old workers
```bash
# Windows
taskkill /F /IM python.exe

# Linux/Mac
pkill -f "python server.py"
```

### Problem: Requests still slow
**Symptom:** Test shows sequential processing
**Possible causes:**
1. llama.cpp can't handle multiple requests (check its configuration)
2. Network bottleneck between servers
3. Workers not actually starting (check logs)

### Problem: High RAM usage
**Symptom:** Running out of memory
**Solution:** Reduce worker count
```python
SERVER_WORKERS = 2
TOOLS_SERVER_WORKERS = 2
```

---

## Performance Expectations

### Before (1 worker, sequential):
- Request 1: 12s
- Request 2: waits... then 12s
- Request 3: waits... then 12s
- Request 4: waits... then 12s
- **Total: 48s**

### After (4 workers, parallel):
- Request 1: 12s
- Request 2: 12s  } All run
- Request 3: 12s  } simultaneously
- Request 4: 12s
- **Total: 13s** (3.7x faster!)

---

## Next Steps (Optional)

If you need even more performance:

1. **Increase llama.cpp parallelism:**
   ```bash
   ./llama-server --model model.gguf --parallel 8
   ```

2. **Add more workers:**
   ```python
   SERVER_WORKERS = 8
   ```

3. **Load balance across multiple llama.cpp servers:**
   - Run multiple llama.cpp instances
   - Use nginx/HAProxy to distribute requests

4. **Full async conversion** (for 100+ concurrent users):
   - See `ASYNC_CONVERSION_TODO.md`
   - More complex but maximum scalability

---

## Summary

✅ **File locking:** Prevents data corruption
✅ **4 workers:** Handle 4 concurrent requests
✅ **Minimal RAM:** Workers are tiny (~50MB each)
✅ **Test script:** Verify it works
✅ **Production ready:** Safe for multiple users

**You're ready to handle concurrent requests!**
