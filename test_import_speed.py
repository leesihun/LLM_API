"""Test script to verify REPL import speed optimizations."""

import sys
import time

# Fix Windows encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'backslashreplace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'backslashreplace')
from pathlib import Path
import tempfile
from backend.tools.python_coder.executor.repl_manager import REPLManager

def test_import_speed():
    """Test that pre-imported libraries are available and fast."""

    print("="*60)
    print("Testing REPL Import Speed Optimization")
    print("="*60)

    # Create temp directory for test
    tmpdir = Path(tempfile.mkdtemp())
    print(f"\nTest directory: {tmpdir}")

    # Create REPL manager
    mgr = REPLManager(timeout=60)
    print("\nCreating REPL session...")

    start_create = time.time()
    repl = mgr.get_or_create('test_session', tmpdir)
    create_time = time.time() - start_create
    print(f"[OK] REPL created in {create_time:.2f}s (includes library pre-loading)")

    # Test 1: Check that pandas is already available
    print("\n" + "="*60)
    print("Test 1: Check pandas is pre-loaded")
    print("="*60)

    code1 = """
import pandas
print(f"Pandas version: {pandas.__version__}")
print(f"pd is available: {'pd' in dir()}")
if 'pd' in dir():
    print(f"pd is pandas: {pd is pandas}")
"""

    start1 = time.time()
    result1 = repl.execute(code1)
    exec_time1 = time.time() - start1

    print(f"Execution time: {exec_time1:.2f}s (should be <0.1s)")
    print(f"Success: {result1['success']}")
    print(f"Output:\n{result1['output']}")

    # Test 2: Check numpy
    print("\n" + "="*60)
    print("Test 2: Check numpy is pre-loaded")
    print("="*60)

    code2 = """
import numpy
print(f"NumPy version: {numpy.__version__}")
print(f"np is available: {'np' in dir()}")
if 'np' in dir():
    arr = np.array([1, 2, 3])
    print(f"Created array with np: {arr}")
"""

    start2 = time.time()
    result2 = repl.execute(code2)
    exec_time2 = time.time() - start2

    print(f"Execution time: {exec_time2:.2f}s (should be <0.1s)")
    print(f"Success: {result2['success']}")
    print(f"Output:\n{result2['output']}")

    # Test 3: Data processing speed (should be fast)
    print("\n" + "="*60)
    print("Test 3: Quick data processing with pre-loaded libs")
    print("="*60)

    code3 = """
# pd and np should already be available!
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(df)
print(f"\\nDataFrame shape: {df.shape}")
"""

    start3 = time.time()
    result3 = repl.execute(code3)
    exec_time3 = time.time() - start3

    print(f"Execution time: {exec_time3:.2f}s (should be <0.1s)")
    print(f"Success: {result3['success']}")
    print(f"Output:\n{result3['output']}")

    # Cleanup
    mgr.cleanup_all()

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"REPL startup (with pre-loading): {create_time:.2f}s")
    print(f"Test 1 (pandas check): {exec_time1:.2f}s")
    print(f"Test 2 (numpy check): {exec_time2:.2f}s")
    print(f"Test 3 (data processing): {exec_time3:.2f}s")

    total_exec = exec_time1 + exec_time2 + exec_time3
    print(f"\nTotal execution time (3 tests): {total_exec:.2f}s")
    print(f"\nOptimization successful: {total_exec < 1.0}")

    print("\n[OK] Test complete!")

if __name__ == "__main__":
    test_import_speed()
