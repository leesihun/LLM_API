"""
Integration test for password validation in API endpoints
"""
import sys
import requests
from fastapi.testclient import TestClient

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def test_signup_with_long_password():
    """Test signup with password exceeding 72 bytes"""
    from backend.api.routes.auth import router
    from fastapi import FastAPI
    
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    
    print("=" * 70)
    print("Testing API Password Validation")
    print("=" * 70 + "\n")
    
    # Test 1: Normal password
    print("[TEST 1] Normal password (should succeed)")
    response = client.post("/api/auth/signup", json={
        "username": "testuser1",
        "password": "SecurePassword123!",
        "role": "user"
    })
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("[OK] Signup successful\n")
    else:
        print(f"[INFO] Response: {response.json()}\n")
    
    # Test 2: Password at limit (72 bytes)
    print("[TEST 2] Password at 72-byte limit (should succeed)")
    response = client.post("/api/auth/signup", json={
        "username": "testuser2",
        "password": "a" * 72,
        "role": "user"
    })
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("[OK] Signup successful\n")
    else:
        print(f"[INFO] Response: {response.json()}\n")
    
    # Test 3: Password over limit (73 bytes) - should fail
    print("[TEST 3] Password over 72-byte limit (should fail with 400)")
    response = client.post("/api/auth/signup", json={
        "username": "testuser3",
        "password": "a" * 73,
        "role": "user"
    })
    print(f"Status: {response.status_code}")
    if response.status_code == 400:
        error_detail = response.json().get("detail", "")
        print(f"[OK] Correctly rejected with error: {error_detail}\n")
        assert "72 bytes" in error_detail, "Error message should mention 72 bytes"
    else:
        print(f"[FAIL] Expected 400, got {response.status_code}\n")
        assert False, "Should have returned 400 error"
    
    # Test 4: Multibyte characters over limit
    print("[TEST 4] Multibyte password over limit (should fail with 400)")
    password_with_unicode = "test" + "가" * 23  # 4 + (3*23) = 73 bytes
    print(f"Password: {len(password_with_unicode)} chars, {len(password_with_unicode.encode('utf-8'))} bytes")
    response = client.post("/api/auth/signup", json={
        "username": "testuser4",
        "password": password_with_unicode,
        "role": "user"
    })
    print(f"Status: {response.status_code}")
    if response.status_code == 400:
        error_detail = response.json().get("detail", "")
        print(f"[OK] Correctly rejected with error: {error_detail}\n")
        assert "72 bytes" in error_detail or "73 bytes" in error_detail
    else:
        print(f"[FAIL] Expected 400, got {response.status_code}\n")
        assert False, "Should have returned 400 error"
    
    # Test 5: Multibyte characters under limit
    print("[TEST 5] Multibyte password under limit (should succeed)")
    password_with_unicode = "test" + "가" * 22  # 4 + (3*22) = 70 bytes
    print(f"Password: {len(password_with_unicode)} chars, {len(password_with_unicode.encode('utf-8'))} bytes")
    response = client.post("/api/auth/signup", json={
        "username": "testuser5",
        "password": password_with_unicode,
        "role": "user"
    })
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("[OK] Signup successful\n")
    else:
        print(f"[INFO] Response: {response.json()}\n")
    
    print("=" * 70)
    print("All API integration tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        test_signup_with_long_password()
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
