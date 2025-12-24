"""
Comprehensive diagnostic for bcrypt module issues
"""
import sys
import traceback

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 70)
print("BCRYPT MODULE DIAGNOSTIC")
print("=" * 70 + "\n")

# Test 1: Basic import
print("[TEST 1] Basic bcrypt import")
print("-" * 70)
try:
    import bcrypt
    print("[OK] bcrypt imported successfully")
    print(f"    Location: {bcrypt.__file__}")
except Exception as e:
    print(f"[ERROR] Failed to import bcrypt: {e}")
    traceback.print_exc()
print()

# Test 2: Check __about__ attribute
print("[TEST 2] Accessing bcrypt.__about__")
print("-" * 70)
try:
    import bcrypt
    about = bcrypt.__about__
    print(f"[OK] bcrypt.__about__ accessible")
    print(f"    Type: {type(about)}")
    print(f"    Value: {about}")
except AttributeError as e:
    print(f"[ERROR] AttributeError: {e}")
    print(f"        This is the error you're experiencing!")
    traceback.print_exc()
except Exception as e:
    print(f"[ERROR] {type(e).__name__}: {e}")
    traceback.print_exc()
print()

# Test 3: Check version attribute
print("[TEST 3] Accessing bcrypt.__version__")
print("-" * 70)
try:
    import bcrypt
    version = bcrypt.__version__
    print(f"[OK] bcrypt.__version__: {version}")
except AttributeError as e:
    print(f"[ERROR] AttributeError: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"[ERROR] {type(e).__name__}: {e}")
    traceback.print_exc()
print()

# Test 4: Check all attributes
print("[TEST 4] All bcrypt attributes")
print("-" * 70)
try:
    import bcrypt
    attrs = dir(bcrypt)
    print(f"[OK] Found {len(attrs)} attributes:")
    
    # Check for metadata attributes
    metadata_attrs = [a for a in attrs if a.startswith('__') and not a.startswith('___')]
    print(f"\nMetadata attributes:")
    for attr in sorted(metadata_attrs):
        try:
            value = getattr(bcrypt, attr)
            print(f"  ✓ {attr}: {type(value).__name__}")
        except Exception as e:
            print(f"  ✗ {attr}: ERROR - {e}")
except Exception as e:
    print(f"[ERROR] {type(e).__name__}: {e}")
    traceback.print_exc()
print()

# Test 5: Passlib integration
print("[TEST 5] Passlib CryptContext with bcrypt")
print("-" * 70)
try:
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    print("[OK] CryptContext created successfully")
    
    # Try hashing
    test_password = "TestPassword123!"
    hashed = pwd_context.hash(test_password)
    print(f"[OK] Password hashed successfully")
    print(f"    Hash: {hashed[:40]}...")
    
    # Try verifying
    verified = pwd_context.verify(test_password, hashed)
    print(f"[OK] Password verified: {verified}")
except Exception as e:
    print(f"[ERROR] {type(e).__name__}: {e}")
    traceback.print_exc()
print()

# Test 6: Import backend.utils.auth
print("[TEST 6] Import backend.utils.auth")
print("-" * 70)
try:
    from backend.utils.auth import hash_password, verify_password
    print("[OK] Successfully imported hash_password and verify_password")
    
    # Try using them
    test_password = "TestPassword123!"
    hashed = hash_password(test_password)
    print(f"[OK] hash_password() works: {hashed[:40]}...")
    
    verified = verify_password(test_password, hashed)
    print(f"[OK] verify_password() works: {verified}")
except Exception as e:
    print(f"[ERROR] {type(e).__name__}: {e}")
    print("\n--- Full Traceback ---")
    traceback.print_exc()
print()

# Test 7: Check for module reload issues
print("[TEST 7] Module reload check")
print("-" * 70)
try:
    import sys
    if 'bcrypt' in sys.modules:
        bcrypt_module = sys.modules['bcrypt']
        print(f"[OK] bcrypt in sys.modules")
        print(f"    Module: {bcrypt_module}")
        print(f"    File: {bcrypt_module.__file__}")
        
        # Check if __about__ is really there
        if hasattr(bcrypt_module, '__about__'):
            print(f"    ✓ Has __about__ attribute")
            about = bcrypt_module.__about__
            print(f"      Type: {type(about)}")
            if hasattr(about, '__file__'):
                print(f"      File: {about.__file__}")
        else:
            print(f"    ✗ Missing __about__ attribute")
            print(f"      Available: {[a for a in dir(bcrypt_module) if a.startswith('__')]}")
    else:
        print("[INFO] bcrypt not yet in sys.modules")
except Exception as e:
    print(f"[ERROR] {type(e).__name__}: {e}")
    traceback.print_exc()
print()

# Test 8: Check Python version compatibility
print("[TEST 8] Python version check")
print("-" * 70)
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

try:
    import bcrypt
    import passlib
    print(f"\nPackage versions:")
    print(f"  bcrypt: {bcrypt.__version__}")
    print(f"  passlib: {passlib.__version__}")
except Exception as e:
    print(f"[ERROR] {type(e).__name__}: {e}")
print()

print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
print("""
If you see "[ERROR] AttributeError: module 'bcrypt' has no attribute '__about__'",
this could be caused by:

1. Corrupted bcrypt installation
   Fix: pip uninstall bcrypt && pip install bcrypt

2. Multiple bcrypt versions installed
   Fix: pip list | grep bcrypt, then uninstall all and reinstall

3. Cached .pyc files
   Fix: python -m py_compile -clean or delete __pycache__ folders

4. Import order issue (rare)
   Fix: Ensure bcrypt is imported before other packages that use it

5. Virtual environment issue
   Fix: Recreate your virtual environment
""")
