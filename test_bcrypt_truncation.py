"""
Test if bcrypt silently truncates passwords over 72 bytes
"""
import sys
import bcrypt
from passlib.context import CryptContext

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 70)
print("Testing Bcrypt Truncation Behavior")
print("=" * 70 + "\n")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Test if passwords that differ only after byte 72 produce same hash
print("TEST: Do passwords differing only after byte 72 produce same hash?")
print("-" * 70)

password1 = "a" * 72 + "b" * 10  # 82 bytes, first 72 are 'a'
password2 = "a" * 72 + "c" * 10  # 82 bytes, first 72 are 'a'
password3 = "a" * 72              # Exactly 72 bytes

print(f"Password 1: {'a'*72+'b'*10} ({len(password1)} bytes)")
print(f"Password 2: {'a'*72+'c'*10} ({len(password2)} bytes)")
print(f"Password 3: {'a'*72} ({len(password3)} bytes)")
print()

# Generate hashes with same salt for comparison
salt = bcrypt.gensalt()

hash1 = bcrypt.hashpw(password1.encode('utf-8'), salt)
hash2 = bcrypt.hashpw(password2.encode('utf-8'), salt)
hash3 = bcrypt.hashpw(password3.encode('utf-8'), salt)

print(f"Hash 1: {hash1}")
print(f"Hash 2: {hash2}")
print(f"Hash 3: {hash3}")
print()

if hash1 == hash2 == hash3:
    print("[CONFIRMED] All three hashes are IDENTICAL!")
    print("            Bcrypt is TRUNCATING passwords to 72 bytes!")
    print()
    print("This means:")
    print("  - Only the first 72 bytes affect the hash")
    print("  - Characters after byte 72 are IGNORED")
    print("  - This is a SECURITY ISSUE if users don't know!")
else:
    print("[INTERESTING] Hashes differ - bcrypt may have changed behavior")
print()

# Verify with passlib
print("-" * 70)
print("Testing password verification with passlib:")
print("-" * 70)

hashed = pwd_context.hash(password1)
print(f"Hashed: {password1[:20]}... ({len(password1)} bytes)")

# Try verifying with different passwords
test_passwords = [
    (password1, "Original password (82 bytes)"),
    (password2, "Different after byte 72 (82 bytes)"),
    (password3, "Truncated to 72 bytes"),
    ("a" * 50, "Only 50 bytes"),
]

for test_pwd, description in test_passwords:
    result = pwd_context.verify(test_pwd, hashed)
    status = "[MATCH]" if result else "[NO MATCH]"
    print(f"{status} {description}: {result}")

print()
print("=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
If passwords 1, 2, and 3 all produce the SAME hash:
  → Bcrypt 4.0.1 is SILENTLY TRUNCATING to 72 bytes
  → No error is raised (behavior change from older versions)
  → This is dangerous - users think long passwords are secure,
    but only first 72 bytes matter!

Our validation is STILL NECESSARY because:
  1. It informs users about the 72-byte limit
  2. It prevents false sense of security
  3. It ensures consistent behavior across versions
  4. Some older bcrypt versions DO raise the error
""")
