# Getting Started in 5 Minutes

## Step 1: Configure Environment (2 minutes)

```bash
# Copy the example configuration
cp .env.example .env
```

Edit `.env` and set these **required** values:

```env
# Generate a secret key (use any random string for testing)
SECRET_KEY=my-super-secret-key-123

# Get your Tavily API key from https://tavily.com
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxxx

# Everything else has sensible defaults you can use as-is
```

## Step 2: Install Ollama (2 minutes)

### Windows
1. Download from https://ollama.ai
2. Install and run Ollama
3. Open PowerShell:
```powershell
ollama pull gpt-oss:20b
```

### Linux/Mac
```bash
curl https://ollama.ai/install.sh | sh
ollama serve &
ollama pull gpt-oss:20b
```

## Step 3: Install Python Dependencies (1 minute)

```bash
pip install -r requirements.txt
```

## Step 4: Run the Server

### Windows
```cmd
run_backend.bat
```

### Linux/Mac
```bash
chmod +x run_backend.sh
./run_backend.sh
```

### Or directly
```bash
python server.py
```

## Step 5: Test It Works

### Option A: Web Browser
Visit: http://localhost:8000/docs

Click "Try it out" on any endpoint

### Option B: Command Line
```bash
# Login
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"guest","password":"guest_test1"}'

# Copy the access_token from response, then:
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss:20b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Option C: Start Frontend + Backend
```bash
# Windows
start_all.bat

# Linux/Mac
./start_all.sh
```

Then visit: http://localhost:3000

---

## That's It! 🎉

Your agentic AI backend is now running!

### What Can It Do?

✅ **Normal Chat**: Just talk naturally
✅ **Web Search**: Ask "What's the latest news about X?"
✅ **Document Q&A**: Upload PDF and ask questions
✅ **Multi-step Reasoning**: Complex queries automatically use tools

### Next Steps

1. **Read the docs**: Open [SETUP.md](SETUP.md) for detailed info
2. **Try the API**: Visit http://localhost:8000/docs
3. **Upload documents**: Use `/api/files/upload` endpoint
4. **Customize**: Edit `.env` to tune behavior

### Default Login

- **Username**: `guest`
- **Password**: `guest_test1`

---

## Quick Troubleshooting

### "Configuration error"
→ Did you copy `.env.example` to `.env`?
→ Did you set `SECRET_KEY` and `TAVILY_API_KEY`?

### "Connection refused"
→ Is Ollama running? Try: `ollama serve`

### "Model not found"
→ Did you pull the model? Try: `ollama pull gpt-oss:20b`

### Port 8000 in use
→ Change `SERVER_PORT=8001` in `.env`

---

## Need Help?

- **Full Setup Guide**: [SETUP.md](SETUP.md)
- **Quick Reference**: [BACKEND_README.md](BACKEND_README.md)
- **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **API Docs**: http://localhost:8000/docs

**Enjoy your agentic AI backend! 🚀**
