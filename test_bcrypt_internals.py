"""
Demonstration of where _bcrypt.hashpw raises the 72-byte error
"""
import sys
import bcrypt
from passlib.context import CryptContext

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 70)
print("Understanding _bcrypt.hashpw() Error")
print("=" * 70 + "\n")

# =============================================================================
# Test 1: Direct bcrypt.hashpw() call
# =============================================================================
print("[TEST 1] Direct bcrypt.hashpw() call")
print("-" * 70)

password_short = "MyPassword123!"
password_long = "a" * 73

print(f"Short password: {len(password_short)} chars, {len(password_short.encode('utf-8'))} bytes")
try:
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_short.encode('utf-8'), salt)
    print(f"[OK] Hashed successfully: {hashed[:30]}...\n")
except Exception as e:
    print(f"[ERROR] {type(e).__name__}: {e}\n")

print(f"Long password: {len(password_long)} chars, {len(password_long.encode('utf-8'))} bytes")
try:
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_long.encode('utf-8'), salt)
    print(f"[OK] Hashed successfully: {hashed[:30]}...\n")
except ValueError as e:
    print(f"[ERROR] ValueError raised by _bcrypt.hashpw(): {e}")
    print(f"        This is the C library rejecting the password!\n")

# =============================================================================
# Test 2: Passlib's pwd_context.hash() call
# =============================================================================
print("[TEST 2] Passlib pwd_context.hash() call")
print("-" * 70)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

print(f"Short password: {len(password_short)} chars")
try:
    hashed = pwd_context.hash(password_short)
    print(f"[OK] Hashed successfully: {hashed[:30]}...\n")
except Exception as e:
    print(f"[ERROR] {type(e).__name__}: {e}\n")

print(f"Long password: {len(password_long)} chars, {len(password_long.encode('utf-8'))} bytes")
try:
    hashed = pwd_context.hash(password_long)
    print(f"[OK] Hashed successfully: {hashed[:30]}...\n")
except ValueError as e:
    print(f"[ERROR] ValueError propagated from _bcrypt.hashpw(): {e}")
    print(f"        Passlib doesn't catch this error, it bubbles up!\n")

# =============================================================================
# Explanation
# =============================================================================
print("=" * 70)
print("WHY DOES THIS HAPPEN?")
print("=" * 70)
print("""
The call stack is:
1. pwd_context.hash(password)              # Passlib wrapper
   └─> 2. bcrypt.hashpw(password, salt)    # Python bcrypt library
       └─> 3. _bcrypt.hashpw(secret, config)  # C extension (compiled code)

The error is raised at level 3 - the C extension!

The C library (_bcrypt) has a hardcoded check:
    if (len(secret) > 72) {
        PyErr_SetString(PyExc_ValueError, 
                       "password cannot get longer than 72 bytes");
        return NULL;
    }

This is a SECURITY FEATURE of bcrypt:
- Bcrypt was designed in 1999 for strong password hashing
- It internally processes passwords in 72-byte blocks
- Longer passwords don't increase security (they get truncated internally)
- The library enforces this limit to prevent misuse

WHY 72 BYTES SPECIFICALLY?
- Bcrypt uses the Blowfish cipher with a 448-bit key
- 448 bits ÷ 8 = 56 bytes
- Plus null terminator and safety margin = 72 bytes maximum

OUR FIX:
Before calling pwd_context.hash(), we check:
    password_bytes = password.encode('utf-8')
    if len(password_bytes) > 72:
        raise ValueError("Password cannot exceed 72 bytes...")

This gives users a BETTER error message BEFORE the C library rejects it.
""")

print("=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
The error occurs at the lowest level (C extension) because:
1. It's a fundamental limitation of the bcrypt algorithm
2. The C code enforces it for security reasons
3. Python libraries (passlib, bcrypt) don't add their own validation
4. Our fix adds Python-level validation with better error messages
""")
