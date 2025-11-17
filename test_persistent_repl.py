"""
Test script for Persistent REPL implementation.

This script tests the new persistent REPL mode for code execution,
which should be significantly faster than spawning new subprocesses.
"""

import time
from pathlib import Path
from backend.tools.python_coder.executor import CodeExecutor

def test_repl_vs_subprocess():
    """Compare performance of REPL vs subprocess execution."""

    print("=" * 70)
    print("PERSISTENT REPL PERFORMANCE TEST")
    print("=" * 70)

    # Test code snippets
    test_codes = [
        "print('Hello from code 1')\nresult = 2 + 2\nprint(f'Result: {result}')",
        "import math\nprint(f'Pi = {math.pi:.4f}')\nprint(f'E = {math.e:.4f}')",
        "nums = [1, 2, 3, 4, 5]\nprint(f'Sum: {sum(nums)}')\nprint(f'Average: {sum(nums)/len(nums)}')",
        "for i in range(3):\n    print(f'Iteration {i+1}')",
        "data = {'a': 1, 'b': 2, 'c': 3}\nfor k, v in data.items():\n    print(f'{k}: {v}')",
    ]

    session_id = "test_session_12345"

    # Test 1: REPL mode (fast)
    print("\n" + "=" * 70)
    print("TEST 1: PERSISTENT REPL MODE (FAST)")
    print("=" * 70)

    executor_repl = CodeExecutor(
        timeout=30,
        max_memory_mb=512,
        execution_base_dir="./data/scratch",
        use_persistent_repl=True
    )

    repl_times = []
    start_total = time.time()

    for i, code in enumerate(test_codes, 1):
        print(f"\n--- Execution {i}/5 (REPL) ---")
        start = time.time()
        result = executor_repl.execute(code, session_id=session_id)
        elapsed = time.time() - start
        repl_times.append(elapsed)

        if result["success"]:
            print(f"[OK] Success in {elapsed:.3f}s")
            print(f"Output: {result['output'][:100]}")
        else:
            print(f"[FAIL] Failed: {result['error']}")

    total_repl_time = time.time() - start_total
    avg_repl_time = sum(repl_times) / len(repl_times)

    # Cleanup REPL
    executor_repl.cleanup_session(session_id)

    # Test 2: Subprocess mode (traditional)
    print("\n" + "=" * 70)
    print("TEST 2: SUBPROCESS MODE (TRADITIONAL)")
    print("=" * 70)

    executor_subprocess = CodeExecutor(
        timeout=30,
        max_memory_mb=512,
        execution_base_dir="./data/scratch",
        use_persistent_repl=False
    )

    subprocess_times = []
    start_total = time.time()

    for i, code in enumerate(test_codes, 1):
        print(f"\n--- Execution {i}/5 (Subprocess) ---")
        start = time.time()
        result = executor_subprocess.execute(code, session_id=session_id + "_subprocess")
        elapsed = time.time() - start
        subprocess_times.append(elapsed)

        if result["success"]:
            print(f"[OK] Success in {elapsed:.3f}s")
            print(f"Output: {result['output'][:100]}")
        else:
            print(f"[FAIL] Failed: {result['error']}")

    total_subprocess_time = time.time() - start_total
    avg_subprocess_time = sum(subprocess_times) / len(subprocess_times)

    # Results comparison
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)
    print(f"\nREPL Mode (Fast):")
    print(f"  Total time: {total_repl_time:.3f}s")
    print(f"  Average per execution: {avg_repl_time:.3f}s")
    print(f"  Individual times: {', '.join(f'{t:.3f}s' for t in repl_times)}")

    print(f"\nSubprocess Mode (Traditional):")
    print(f"  Total time: {total_subprocess_time:.3f}s")
    print(f"  Average per execution: {avg_subprocess_time:.3f}s")
    print(f"  Individual times: {', '.join(f'{t:.3f}s' for t in subprocess_times)}")

    speedup = total_subprocess_time / total_repl_time
    print(f"\n>>> SPEEDUP: {speedup:.2f}x faster with REPL mode!")
    print(f"   Time saved: {total_subprocess_time - total_repl_time:.3f}s ({(1 - 1/speedup)*100:.1f}% reduction)")

    # Expected savings in real workflow
    print("\n" + "=" * 70)
    print("PROJECTED SAVINGS IN REAL WORKFLOW")
    print("=" * 70)
    print(f"\nFor 5 retry attempts (worst case):")
    print(f"  Subprocess mode: {avg_subprocess_time * 5:.2f}s")
    print(f"  REPL mode: {avg_repl_time * 5:.2f}s")
    print(f"  Savings: {(avg_subprocess_time - avg_repl_time) * 5:.2f}s per task")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


def test_repl_error_handling():
    """Test REPL error handling and recovery."""

    print("\n" + "=" * 70)
    print("REPL ERROR HANDLING TEST")
    print("=" * 70)

    executor = CodeExecutor(
        timeout=30,
        max_memory_mb=512,
        execution_base_dir="./data/scratch",
        use_persistent_repl=True
    )

    session_id = "test_error_session"

    # Test 1: Success
    print("\n--- Test 1: Successful execution ---")
    result = executor.execute("print('Test 1 success')", session_id=session_id)
    print(f"Result: {'[OK] Success' if result['success'] else '[FAIL] Failed'}")

    # Test 2: Error (should not crash REPL)
    print("\n--- Test 2: Code with error ---")
    result = executor.execute("1 / 0  # Division by zero", session_id=session_id)
    print(f"Result: {'[OK] Success' if result['success'] else '[EXPECTED] Expected failure'}")
    print(f"Error: {result['error'][:100] if result['error'] else 'None'}")

    # Test 3: Recovery (REPL should still work)
    print("\n--- Test 3: Recovery after error ---")
    result = executor.execute("print('Test 3 - REPL recovered!')", session_id=session_id)
    print(f"Result: {'[OK] REPL recovered!' if result['success'] else '[FAIL] REPL crashed'}")

    # Cleanup
    executor.cleanup_session(session_id)
    print("\n[OK] Error handling test complete!")


if __name__ == "__main__":
    # Run performance test
    test_repl_vs_subprocess()

    # Run error handling test
    test_repl_error_handling()
