# Quick Fix: Ollama Access Denied on Linux

## The Problem

```
Error calling Ollama: Access denied from URL 127.0.0.1:11434/api/generate
```

## The Solution (30 seconds)

Run these commands on your **Linux machine where Ollama is installed**:

```bash
# Stop Ollama if it's running
killall ollama

# Start Ollama with network binding
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

Keep this terminal window open. Ollama is now accessible!

## Make It Permanent

Add this to your `~/.bashrc`:

```bash
echo 'export OLLAMA_HOST=0.0.0.0:11434' >> ~/.bashrc
source ~/.bashrc
```

Now you can just run `ollama serve` normally.

## Verify It Works

```bash
# Should return JSON with your models
curl http://127.0.0.1:11434/api/tags
```

## Then Start Your API Server

```bash
cd /path/to/LLM_API
bash start_servers.sh
```

Done! ✅

---

**Still having issues?** Check [LINUX_DEPLOYMENT.md](LINUX_DEPLOYMENT.md) for detailed troubleshooting.
