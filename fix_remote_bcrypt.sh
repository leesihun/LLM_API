#!/bin/bash
# Fix bcrypt 5.0.0 issue on remote system
# Run this script on the remote server

echo "========================================================================"
echo "Bcrypt Remote System Fix Script"
echo "========================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check current bcrypt version
echo "1. Checking current bcrypt version..."
CURRENT_VERSION=$(python -c "import bcrypt; print(bcrypt.__version__)" 2>&1)

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Current bcrypt version: ${CURRENT_VERSION}"
else
    echo -e "${RED}✗${NC} Failed to detect bcrypt version"
    echo "Error: $CURRENT_VERSION"
fi

echo ""

# Check if version is 5.0.0
if [[ "$CURRENT_VERSION" == "5.0.0" ]]; then
    echo -e "${YELLOW}⚠${NC}  Bcrypt 5.0.0 detected - this version has breaking changes"
    echo "   Will downgrade to stable version 4.0.1"
    NEEDS_FIX=true
else
    echo -e "${GREEN}✓${NC} Bcrypt version looks OK"
    NEEDS_FIX=false
fi

echo ""

# Ask for confirmation
if [ "$NEEDS_FIX" = true ]; then
    echo "2. Ready to fix bcrypt installation"
    echo "   This will:"
    echo "   - Uninstall bcrypt 5.0.0"
    echo "   - Clear pip cache"
    echo "   - Install bcrypt 4.0.1"
    echo ""
    read -p "Continue? (y/n): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "3. Uninstalling bcrypt 5.0.0..."
        pip uninstall bcrypt -y
        echo -e "${GREEN}✓${NC} Uninstalled"
        
        echo ""
        echo "4. Clearing pip cache..."
        pip cache purge
        echo -e "${GREEN}✓${NC} Cache cleared"
        
        echo ""
        echo "5. Installing bcrypt 4.0.1..."
        pip install bcrypt==4.0.1
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓${NC} Installed bcrypt 4.0.1"
        else
            echo -e "${RED}✗${NC} Failed to install bcrypt 4.0.1"
            exit 1
        fi
        
        echo ""
        echo "6. Verifying installation..."
        NEW_VERSION=$(python -c "import bcrypt; print(bcrypt.__version__)" 2>&1)
        
        if [[ "$NEW_VERSION" == "4.0.1" ]]; then
            echo -e "${GREEN}✓${NC} bcrypt 4.0.1 installed successfully"
        else
            echo -e "${RED}✗${NC} Unexpected version: $NEW_VERSION"
            exit 1
        fi
    else
        echo ""
        echo "Installation cancelled"
        exit 0
    fi
else
    echo "2. No fix needed - bcrypt version is compatible"
fi

echo ""
echo "========================================================================"
echo "Testing Authentication"
echo "========================================================================"
echo ""

# Test authentication
echo "Running authentication test..."
python -c "
from backend.utils.auth import hash_password, verify_password

try:
    # Test 1: Hash a password
    test_pwd = 'TestPassword123!'
    hashed = hash_password(test_pwd)
    print(f'✓ Hash password works: {hashed[:30]}...')
    
    # Test 2: Verify password
    verified = verify_password(test_pwd, hashed)
    if verified:
        print(f'✓ Verify password works: {verified}')
    else:
        print(f'✗ Verify password failed!')
        exit(1)
    
    # Test 3: Test 72-byte limit
    try:
        hash_password('a' * 73)
        print('✗ Validation failed - should reject 73-byte password')
        exit(1)
    except ValueError as e:
        print(f'✓ Validation works: Correctly rejected >72 bytes')
    
    print('')
    print('✓✓✓ All authentication tests passed!')
    
except Exception as e:
    print(f'✗ Authentication test failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo -e "${GREEN}SUCCESS!${NC} Bcrypt is working correctly"
    echo "========================================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Update requirements.txt to pin bcrypt==4.0.1"
    echo "  2. Restart your application"
    echo "  3. Test login/signup endpoints"
    echo ""
else
    echo ""
    echo "========================================================================"
    echo -e "${RED}FAILED!${NC} Authentication tests did not pass"
    echo "========================================================================"
    echo ""
    echo "Please check:"
    echo "  - Python environment is correct"
    echo "  - All dependencies are installed"
    echo "  - Application code is up to date"
    echo ""
    exit 1
fi
