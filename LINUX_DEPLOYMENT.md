# Linux Deployment Guide

## Issue: "Access denied from URL 127.0.0.1:11434/api/generate"

This error occurs on Linux when Ollama is configured to only accept connections from localhost on the loopback interface, but the Python client is trying to connect from a different network context.

## Solution

### Option 1: Configure Ollama to Listen on All Interfaces (Recommended for Local Deployment)

This allows Ollama to accept connections from any network interface.

**Permanent Solution:**

1. Set the environment variable system-wide:
```bash
echo 'export OLLAMA_HOST=0.0.0.0:11434' >> ~/.bashrc
source ~/.bashrc
```

2. Restart Ollama:
```bash
# Stop Ollama if running
killall ollama

# Start Ollama with the new configuration
ollama serve
```

**Temporary Solution (Single Session):**

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

### Option 2: Use Systemd Service (Best for Production)

1. Create a systemd service file:
```bash
sudo nano /etc/systemd/system/ollama.service
```

2. Add the following content:
```ini
[Unit]
Description=Ollama Service
After=network.target

[Service]
Type=simple
User=your-username
Environment="OLLAMA_HOST=0.0.0.0:11434"
ExecStart=/usr/local/bin/ollama serve
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

3. Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl start ollama
sudo systemctl status ollama
```

### Option 3: Firewall Configuration (If Using Remote Ollama Server)

If Ollama is running on a different machine:

1. Update `config.py`:
```python
OLLAMA_HOST = "http://<ollama-server-ip>:11434"
```

2. Ensure firewall allows connections:
```bash
# On the Ollama server
sudo ufw allow 11434/tcp
```

## Verification

Test that Ollama is accessible:

```bash
# From the same machine
curl http://127.0.0.1:11434/api/tags

# From local network (if configured for 0.0.0.0)
curl http://localhost:11434/api/tags

# From remote machine
curl http://<server-ip>:11434/api/tags
```

## Starting the LLM API Server on Linux

### 1. Install Dependencies

```bash
cd /path/to/LLM_API
pip install -r requirements.txt
```

### 2. Configure Ollama (Choose One Option Above)

Make sure Ollama is running and accessible.

### 3. Update Configuration

Edit `config.py`:
```python
# If Ollama is on the same machine with 0.0.0.0 binding
OLLAMA_HOST = "http://127.0.0.1:11434"

# If Ollama is on a different machine
OLLAMA_HOST = "http://<ollama-server-ip>:11434"

# Choose your model
OLLAMA_MODEL = "gpt-oss:120b"  # or any model you have installed

# For python_coder tool (OpenInterpreter mode)
TOOL_MODELS = {
    "python_coder": "gemma3:1b",  # Use smaller model for faster execution
}
```

### 4. Start the Servers

```bash
# Make sure the start script is executable
chmod +x start_servers.sh

# Start both servers
bash start_servers.sh
```

Or start them separately in different terminals:

```bash
# Terminal 1: Tools server (must start first)
python tools_server.py

# Terminal 2: Main API server
python server.py
```

### 5. Verify Everything Works

```bash
# Check Ollama
curl http://127.0.0.1:11434/api/tags

# Check Tools API
curl http://localhost:10006/health

# Check Main API
curl http://localhost:10007/health
```

## Troubleshooting

### Error: "Access denied"

- **Cause**: Ollama is not configured to accept network connections
- **Solution**: Follow Option 1 or Option 2 above

### Error: "Connection refused"

- **Cause**: Ollama is not running
- **Solution**: Start Ollama with `ollama serve`

### Error: "No such model"

- **Cause**: The model specified in config doesn't exist
- **Solution**:
  ```bash
  # List available models
  ollama list

  # Pull a model if needed
  ollama pull gemma3:1b
  ```

### OpenInterpreter Hangs

- **Cause**: Using large models takes time
- **Solution**:
  - Use smaller models like `gemma3:1b` or `deepseek-r1:1.5b`
  - Increase timeout in config: `PYTHON_CODER_TIMEOUT = 600`

## Security Considerations

### For Production Deployments:

1. **Do NOT use `0.0.0.0` on public-facing servers** - it exposes Ollama to the internet
2. Instead, use Docker networking or bind to specific internal IPs:
   ```bash
   OLLAMA_HOST=192.168.1.100:11434 ollama serve
   ```
3. Use a reverse proxy (nginx) with authentication
4. Configure firewall rules to restrict access
5. Change default credentials in `config.py`:
   ```python
   JWT_SECRET_KEY = "your-very-secure-random-key"
   DEFAULT_ADMIN_PASSWORD = "strong-password"
   ```

## Performance Optimization

### For Better Performance on Linux:

1. **Use smaller models for tools**:
   ```python
   TOOL_MODELS = {
       "python_coder": "gemma3:1b",  # Fast execution
   }
   ```

2. **Enable GPU acceleration** (if available):
   ```bash
   # Install CUDA support for Ollama
   # Ollama automatically uses GPU if available
   ```

3. **Adjust timeouts** based on your hardware:
   ```python
   PYTHON_CODER_TIMEOUT = 300  # Lower for fast systems
   REACT_MAX_ITERATIONS = 5    # Lower for faster responses
   ```

## Common Linux-Specific Issues

### Permission Denied for Data Directories

```bash
# Fix permissions
sudo chown -R $USER:$USER data/
chmod -R 755 data/
```

### Port Already in Use

```bash
# Find what's using the port
sudo lsof -i :10006
sudo lsof -i :10007

# Kill the process or change ports in config.py
```

### Python Version Issues

```bash
# Make sure you're using Python 3.8+
python --version

# Use virtual environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
