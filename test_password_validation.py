"""
Test password validation for bcrypt 72-byte limit
"""
import sys
from backend.utils.auth import hash_password, verify_password

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def test_normal_password():
    """Test normal password hashing"""
    password = "MySecurePassword123!"
    print(f"[OK] Testing normal password: {len(password)} chars, {len(password.encode('utf-8'))} bytes")
    
    hashed = hash_password(password)
    print(f"[OK] Hashing successful")
    
    assert verify_password(password, hashed)
    print(f"[OK] Verification successful\n")


def test_long_password_under_limit():
    """Test password near but under 72 bytes"""
    password = "a" * 72  # Exactly 72 bytes
    print(f"[OK] Testing password at limit: {len(password)} chars, {len(password.encode('utf-8'))} bytes")
    
    hashed = hash_password(password)
    print(f"[OK] Hashing successful")
    
    assert verify_password(password, hashed)
    print(f"[OK] Verification successful\n")


def test_password_over_limit():
    """Test password over 72 bytes - should raise ValueError"""
    password = "a" * 73  # 73 bytes - over limit
    print(f"[TEST] Testing password over limit: {len(password)} chars, {len(password.encode('utf-8'))} bytes")
    
    try:
        hash_password(password)
        print("[FAIL] ERROR: Should have raised ValueError!")
        assert False, "Expected ValueError"
    except ValueError as e:
        print(f"[OK] Correctly raised ValueError: {e}\n")


def test_multibyte_characters():
    """Test password with unicode characters"""
    # Korean characters are typically 3 bytes each
    password = "test" + "가" * 23  # 4 + (3*23) = 73 bytes, should fail
    print(f"[TEST] Testing multibyte password: {len(password)} chars, {len(password.encode('utf-8'))} bytes")
    
    try:
        hash_password(password)
        print("[FAIL] ERROR: Should have raised ValueError!")
        assert False, "Expected ValueError"
    except ValueError as e:
        print(f"[OK] Correctly raised ValueError: {e}\n")


def test_multibyte_under_limit():
    """Test password with unicode characters under limit"""
    password = "test" + "가" * 22  # 4 + (3*22) = 70 bytes, should succeed
    print(f"[OK] Testing multibyte password: {len(password)} chars, {len(password.encode('utf-8'))} bytes")
    
    hashed = hash_password(password)
    print(f"[OK] Hashing successful")
    
    assert verify_password(password, hashed)
    print(f"[OK] Verification successful\n")


if __name__ == "__main__":
    print("=" * 70)
    print("Password Validation Tests - Bcrypt 72-byte Limit")
    print("=" * 70 + "\n")
    
    test_normal_password()
    test_long_password_under_limit()
    test_password_over_limit()
    test_multibyte_under_limit()
    test_multibyte_characters()
    
    print("=" * 70)
    print("All tests passed!")
    print("=" * 70)
