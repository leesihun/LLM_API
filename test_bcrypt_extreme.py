"""
Test extreme password lengths to trigger _bcrypt error
"""
import sys
import bcrypt
from passlib.context import CryptContext

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 70)
print("Testing Extreme Password Lengths")
print("=" * 70 + "\n")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

test_cases = [
    (50, "a" * 50),
    (72, "a" * 72),
    (73, "a" * 73),
    (100, "a" * 100),
    (200, "a" * 200),
    (1000, "a" * 1000),
]

for length, password in test_cases:
    byte_length = len(password.encode('utf-8'))
    print(f"Testing password: {length} chars, {byte_length} bytes")
    
    # Test with passlib
    try:
        hashed = pwd_context.hash(password)
        print(f"  [PASSLIB] Success: {hashed[:40]}...")
    except ValueError as e:
        print(f"  [PASSLIB] ValueError: {e}")
    except Exception as e:
        print(f"  [PASSLIB] {type(e).__name__}: {e}")
    
    # Test with direct bcrypt
    try:
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        print(f"  [BCRYPT]  Success: {hashed[:40]}...")
    except ValueError as e:
        print(f"  [BCRYPT]  ValueError: {e}")
    except Exception as e:
        print(f"  [BCRYPT]  {type(e).__name__}: {e}")
    
    print()

# Test with multibyte characters
print("=" * 70)
print("Testing with multibyte characters (Korean)")
print("=" * 70 + "\n")

multibyte_tests = [
    ("가" * 24, "24 Korean chars"),  # 72 bytes
    ("가" * 25, "25 Korean chars"),  # 75 bytes
    ("가" * 100, "100 Korean chars"), # 300 bytes
]

for password, description in multibyte_tests:
    byte_length = len(password.encode('utf-8'))
    print(f"Testing: {description}, {byte_length} bytes")
    
    try:
        hashed = pwd_context.hash(password)
        print(f"  [PASSLIB] Success: {hashed[:40]}...")
    except ValueError as e:
        print(f"  [PASSLIB] ValueError: {e}")
    except Exception as e:
        print(f"  [PASSLIB] {type(e).__name__}: {e}")
    print()
