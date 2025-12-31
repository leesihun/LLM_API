# User Management Guide

## Quick Reference

### Default Admin User
```
Username: admin
Password: administrator
```
Created automatically on first startup.

---

## Method 1: API Signup (Self-Service)

**Best for:** Users creating their own accounts

### Python:
```python
import requests

response = requests.post(
    "http://localhost:10007/api/auth/signup",
    json={
        "username": "newuser",
        "password": "securepass123"
    }
)

print(response.json())
```

### curl:
```bash
curl -X POST http://localhost:10007/api/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"username": "newuser", "password": "securepass123"}'
```

### Response:
```json
{
  "username": "newuser",
  "role": "user"
}
```

---

## Method 2: Admin API (Bulk Creation)

**Best for:** Admins creating multiple users

### Run the script:
```bash
python create_users.py
```

### Or manually:

1. **Login as admin:**
```python
import requests

response = requests.post(
    "http://localhost:10007/api/auth/login",
    data={"username": "admin", "password": "administrator"}
)

token = response.json()["access_token"]
```

2. **Create user:**
```python
headers = {"Authorization": f"Bearer {token}"}

response = requests.post(
    "http://localhost:10007/api/admin/users",
    json={
        "username": "newuser",
        "password": "securepass",
        "role": "user"  # or "admin"
    },
    headers=headers
)
```

3. **List all users:**
```python
response = requests.get(
    "http://localhost:10007/api/admin/users",
    headers=headers
)

print(response.json())
```

---

## Method 3: Direct Database (Advanced)

**Best for:** When API is not running or for initial setup

### Interactive mode:
```bash
python create_user_direct.py
```

### Programmatic:
```python
from backend.core.database import db
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
password_hash = pwd_context.hash("mypassword")

db.create_user("username", password_hash, "user")
```

---

## Available Admin Endpoints

All require admin authentication (Authorization: Bearer {token})

### Create User
```
POST /api/admin/users
Body: {"username": "...", "password": "...", "role": "user|admin"}
```

### List All Users
```
GET /api/admin/users
```

### Delete User
```
DELETE /api/admin/users/{username}
```

### Get User Info
```
GET /api/admin/users/{username}
```

---

## User Roles

### `user` (default)
- Access to chat API
- Manage own sessions
- Upload files
- Use RAG collections
- Use tools (websearch, python_coder, etc.)

### `admin`
- All user permissions
- Create/delete users
- View all users
- Access admin endpoints

---

## Password Requirements

⚠️ **Important:** Passwords have a **72-byte limit** (bcrypt restriction)

- ASCII characters: ~72 characters max
- Emoji/Unicode: Much less (emoji can be 4+ bytes each)
- See [BCRYPT_PASSWORD_FIX.md](BCRYPT_PASSWORD_FIX.md) for details

**Safe password examples:**
- ✅ `MySecurePassword123!` (21 bytes)
- ✅ `correct-horse-battery-staple` (29 bytes)
- ⚠️ `🔐🔑🗝️🔓🔒🔐🔑🗝️🔓🔒🔐🔑🗝️🔓🔒🔐🔑🗝️` (72 bytes, but only 18 emoji!)

---

## Example: Creating Test Users

```python
# create_test_users.py
import requests

API = "http://localhost:10007"

# Login as admin
login = requests.post(f"{API}/api/auth/login",
    data={"username": "admin", "password": "administrator"})
token = login.json()["access_token"]

headers = {"Authorization": f"Bearer {token}"}

# Create test users
users = [
    ("alice", "alice123", "user"),
    ("bob", "bob123", "user"),
    ("charlie", "charlie123", "user"),
]

for username, password, role in users:
    resp = requests.post(
        f"{API}/api/admin/users",
        json={"username": username, "password": password, "role": role},
        headers=headers
    )
    print(f"Created {username}: {resp.status_code}")
```

---

## Troubleshooting

### "User already exists"
- Username must be unique
- Try a different username

### "Password too long"
- Password exceeds 72 bytes
- Use shorter password or fewer Unicode characters

### "Unauthorized" when creating users
- Only admins can use `/api/admin/users` endpoint
- Regular users must use `/api/auth/signup`
- Check your JWT token

### Can't login after creating user
- Verify password was typed correctly
- Check database: `python create_user_direct.py` and list users
- Password is case-sensitive

---

## Scripts Summary

| Script | Purpose | When to Use |
|--------|---------|-------------|
| [create_users.py](create_users.py) | Bulk create users via API | When servers are running |
| [create_user_direct.py](create_user_direct.py) | Direct database access | When API is down or for setup |
| API `/api/auth/signup` | Self-service signup | For end users |
| API `/api/admin/users` | Admin user management | For programmatic access |

---

## Security Notes

1. **Change default admin password** in production:
   - Edit `config.py`: `DEFAULT_ADMIN_PASSWORD`
   - Or delete admin user and recreate with strong password

2. **Use HTTPS** in production:
   - JWT tokens are bearer tokens
   - Unencrypted HTTP exposes tokens

3. **Password storage:**
   - Passwords are hashed with bcrypt (industry standard)
   - Original passwords are never stored
   - Hash verification is slow by design (prevents brute force)

4. **JWT tokens:**
   - Default expiration: 7 days (configurable in `config.py`)
   - Tokens stored client-side only
   - Server restart doesn't invalidate tokens (uses SECRET_KEY)
