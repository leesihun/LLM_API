#!/usr/bin/env python3
"""Test Unicode character validation in CodeSandbox."""

from backend.tools.code_sandbox import CodeSandbox

def test_validation():
    sandbox = CodeSandbox()

    # Test 1: Middle dot (U+00B7) - SHOULD FAIL
    print("Test 1: Middle dot (U+00B7)")
    code1 = "x = 5 \u00B7 3"
    valid, issues = sandbox.validate_imports(code1)
    print(f"  Code: {repr(code1)}")
    print(f"  Valid: {valid}")
    print(f"  Expected: False")
    print(f"  Issues: {issues}")
    assert not valid, "Middle dot should be rejected"
    print("  ✓ PASSED\n")

    # Test 2: Multiplication sign (U+00D7) - SHOULD FAIL
    print("Test 2: Multiplication sign (U+00D7)")
    code2 = "x = 5 \u00D7 3"
    valid, issues = sandbox.validate_imports(code2)
    print(f"  Code: {repr(code2)}")
    print(f"  Valid: {valid}")
    print(f"  Expected: False")
    print(f"  Issues: {issues}")
    assert not valid, "Multiplication sign should be rejected"
    print("  ✓ PASSED\n")

    # Test 3: Normal Python - SHOULD PASS
    print("Test 3: Normal Python code")
    code3 = "x = 5 * 3"
    valid, issues = sandbox.validate_imports(code3)
    print(f"  Code: {repr(code3)}")
    print(f"  Valid: {valid}")
    print(f"  Expected: True")
    print(f"  Issues: {issues}")
    assert valid, "Normal code should pass"
    print("  ✓ PASSED\n")

    # Test 4: Unicode in strings - SHOULD PASS
    print("Test 4: Unicode in strings (Chinese)")
    code4 = 'msg = "你好世界"'
    valid, issues = sandbox.validate_imports(code4)
    print(f"  Code: {repr(code4)}")
    print(f"  Valid: {valid}")
    print(f"  Expected: True")
    print(f"  Issues: {issues}")
    assert valid, "Unicode in strings should pass"
    print("  ✓ PASSED\n")

    # Test 5: Smart quotes (U+201C, U+201D) - SHOULD FAIL
    print("Test 5: Smart quotes (U+201C, U+201D)")
    code5 = "print(\u201CHello\u201D)"
    valid, issues = sandbox.validate_imports(code5)
    print(f"  Code: {repr(code5)}")
    print(f"  Valid: {valid}")
    print(f"  Expected: False")
    print(f"  Issues: {issues}")
    assert not valid, "Smart quotes should be rejected"
    print("  ✓ PASSED\n")

    # Test 6: Non-breaking space (U+00A0) - SHOULD FAIL
    print("Test 6: Non-breaking space (U+00A0)")
    code6 = "x\u00A0=\u00A05"
    valid, issues = sandbox.validate_imports(code6)
    print(f"  Code: {repr(code6)}")
    print(f"  Valid: {valid}")
    print(f"  Expected: False")
    print(f"  Issues: {issues}")
    assert not valid, "Non-breaking space should be rejected"
    print("  ✓ PASSED\n")

    # Test 7: Blocked import - SHOULD FAIL
    print("Test 7: Blocked import (subprocess)")
    code7 = "import subprocess"
    valid, issues = sandbox.validate_imports(code7)
    print(f"  Code: {repr(code7)}")
    print(f"  Valid: {valid}")
    print(f"  Expected: False")
    print(f"  Issues: {issues}")
    assert not valid, "Blocked imports should be rejected"
    print("  ✓ PASSED\n")

    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

if __name__ == "__main__":
    test_validation()
