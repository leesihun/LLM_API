# Deployment Guide: Fix Bcrypt on Remote System

## 🎯 Quick Fix for Remote System

Your remote system has **bcrypt 5.0.0** which has breaking changes. Here's how to fix it:

## 🚀 Method 1: Automated Script (Easiest)

### Step 1: Upload fix script to remote
```bash
# On local machine
scp fix_remote_bcrypt.sh user@remote-server:/path/to/LLM_API/
scp requirements.txt user@remote-server:/path/to/LLM_API/
```

### Step 2: Run fix script on remote
```bash
# SSH into remote
ssh user@remote-server
cd /path/to/LLM_API

# Make executable and run
chmod +x fix_remote_bcrypt.sh
./fix_remote_bcrypt.sh
```

The script will:
- ✅ Detect bcrypt 5.0.0
- ✅ Downgrade to 4.0.1
- ✅ Test authentication
- ✅ Verify everything works

## 🔧 Method 2: Manual Fix

### On Remote Server:

```bash
# 1. SSH into remote
ssh user@remote-server
cd /path/to/LLM_API

# 2. Stop application (adjust command for your setup)
sudo systemctl stop llm-api
# OR: pkill -f "uvicorn main:app"
# OR: docker-compose down

# 3. Fix bcrypt
pip uninstall bcrypt -y
pip cache purge
pip install bcrypt==4.0.1

# 4. Verify version
python -c "import bcrypt; print(f'bcrypt version: {bcrypt.__version__}')"
# Should output: bcrypt version: 4.0.1

# 5. Test authentication
python -c "
from backend.utils.auth import hash_password, verify_password
h = hash_password('test')
print(f'✓ Works: {verify_password(\"test\", h)}')
"

# 6. Restart application
sudo systemctl start llm-api
# OR: docker-compose up -d
# OR: nohup uvicorn main:app --host 0.0.0.0 --port 8000 &
```

## 📦 Method 3: Update From requirements.txt

### Step 1: Push updated requirements.txt
```bash
# On local machine (already done)
git add requirements.txt
git commit -m "Pin bcrypt to 4.0.1 for compatibility"
git push
```

### Step 2: Pull and install on remote
```bash
# On remote server
cd /path/to/LLM_API
git pull

# Stop application
sudo systemctl stop llm-api

# Install fixed versions
pip install -r requirements.txt --upgrade

# Restart application
sudo systemctl start llm-api
```

## ✅ Verification Steps

### 1. Check bcrypt version
```bash
python -c "import bcrypt; print(bcrypt.__version__)"
# Expected: 4.0.1
```

### 2. Test authentication module
```bash
python -c "
from backend.utils.auth import hash_password, verify_password
try:
    h = hash_password('testpassword')
    v = verify_password('testpassword', h)
    print(f'✅ AUTH WORKS: verify={v}')
except Exception as e:
    print(f'❌ AUTH FAILED: {e}')
"
```

### 3. Test password validation
```bash
python -c "
from backend.utils.auth import hash_password
try:
    hash_password('a' * 73)  # Should fail
    print('❌ VALIDATION BROKEN')
except ValueError as e:
    print(f'✅ VALIDATION WORKS: {e}')
"
```

### 4. Test API endpoints
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test signup (should reject long password)
curl -X POST http://localhost:8000/api/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","password":"'$(python -c 'print("a"*73)')'","role":"user"}'
# Should return 400 with clear error message

# Test signup (should work with normal password)
curl -X POST http://localhost:8000/api/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser2","password":"SecurePass123!","role":"user"}'
# Should return 200 with token
```

## 🐋 Docker Deployment

If using Docker, update your Dockerfile or docker-compose.yml:

### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements with pinned versions
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Verify bcrypt version
RUN python -c "import bcrypt; assert bcrypt.__version__ == '4.0.1', f'Wrong bcrypt version: {bcrypt.__version__}'"

# Copy application
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Rebuild and redeploy
```bash
# On remote server
cd /path/to/LLM_API
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Check logs
docker-compose logs -f
```

## 🔄 Environment Sync

To prevent future version mismatches:

### 1. Pin all critical dependencies
```txt
# requirements.txt
bcrypt==4.0.1          # Password hashing
passlib==1.7.4         # Password validation
python-jose==3.3.0     # JWT tokens
fastapi==0.104.1       # Web framework
uvicorn==0.24.0        # ASGI server
```

### 2. Use pip-tools
```bash
# Install pip-tools
pip install pip-tools

# Create requirements.in (high-level deps)
echo "bcrypt==4.0.1" > requirements.in
echo "passlib==1.7.4" >> requirements.in
echo "fastapi" >> requirements.in

# Generate pinned requirements.txt
pip-compile requirements.in

# Install exact versions
pip-sync requirements.txt
```

### 3. Test in staging first
```bash
# Always test on staging before production
# staging server:
git pull
pip install -r requirements.txt
./run_tests.sh
```

## 📊 Version Matrix

| Environment | bcrypt | passlib | Status |
|-------------|--------|---------|--------|
| Local Dev   | 4.0.1  | 1.7.4   | ✅ Working |
| Remote/Prod | 5.0.0  | 1.7.4   | ❌ Broken |
| After Fix   | 4.0.1  | 1.7.4   | ✅ Fixed |

## 🆘 Troubleshooting

### Error: "Permission denied"
```bash
# Use sudo or activate venv
sudo pip install bcrypt==4.0.1
# OR
source venv/bin/activate
pip install bcrypt==4.0.1
```

### Error: "pip: command not found"
```bash
# Use python -m pip
python -m pip install bcrypt==4.0.1
```

### Error: "No module named 'backend'"
```bash
# Make sure you're in project root
cd /path/to/LLM_API
export PYTHONPATH=$PWD:$PYTHONPATH
python -c "from backend.utils.auth import hash_password"
```

### Error persists after fix
```bash
# Full cleanup
pip uninstall bcrypt passlib python-jose -y
pip cache purge
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
pip install bcrypt==4.0.1 passlib==1.7.4 python-jose
```

## 📞 Support Checklist

If still having issues, collect this info:

```bash
# System info
python --version
pip --version
which python
pip show bcrypt passlib

# Test output
python diagnose_bcrypt_issue.py > diagnostic.txt

# Application logs
tail -n 100 /var/log/llm-api.log  # adjust path
# OR
journalctl -u llm-api -n 100
# OR
docker logs llm-api --tail 100
```

---

## 🎉 Success Indicators

After fix, you should see:
- ✅ `bcrypt.__version__` = `4.0.1`
- ✅ Authentication tests pass
- ✅ Password validation works (rejects >72 bytes)
- ✅ API endpoints respond correctly
- ✅ Login/signup works in application
- ✅ No errors in logs

**Status:** 🟢 Ready to deploy!
