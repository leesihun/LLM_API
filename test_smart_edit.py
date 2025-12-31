"""
Test script for python_coder smart edit functionality
"""
import sys
sys.path.insert(0, '.')

from tools.python_coder.tool import PythonCoderTool
import config

# Enable smart edit
config.PYTHON_CODER_SMART_EDIT = True

def test_scenario_1_no_existing_files():
    """Test: No existing files - should create new file"""
    print("\n" + "="*80)
    print("TEST 1: No existing files")
    print("="*80)

    tool = PythonCoderTool(session_id="test_session_1")
    tool.clear_workspace()

    code = """
import pandas as pd

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
print(df)
"""

    result = tool.execute(code)
    print(f"Success: {result['success']}")
    print(f"Files created: {list(result['files'].keys())}")
    print(f"Output:\n{result['stdout']}")

    tool.clear_workspace()


def test_scenario_2_edit_existing():
    """Test: Existing file - should merge/edit"""
    print("\n" + "="*80)
    print("TEST 2: Edit existing file")
    print("="*80)

    tool = PythonCoderTool(session_id="test_session_2")
    tool.clear_workspace()

    # First execution: Create initial analysis
    code1 = """
import pandas as pd

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
print("Data loaded:")
print(df.head())
"""

    print("\n--- First execution (create initial file) ---")
    result1 = tool.execute(code1)
    print(f"Success: {result1['success']}")
    print(f"Files: {list(result1['files'].keys())}")

    # Second execution: Add visualization (should edit existing)
    code2 = """
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(df['a'], df['b'], marker='o')
plt.xlabel('Column A')
plt.ylabel('Column B')
plt.title('A vs B')
plt.savefig('plot.png')
print("Plot saved to plot.png")
"""

    print("\n--- Second execution (should merge with existing) ---")
    result2 = tool.execute(code2)
    print(f"Success: {result2['success']}")
    print(f"Files: {list(result2['files'].keys())}")
    print(f"Output:\n{result2['stdout']}")

    # Check if plot was created
    if 'plot.png' in result2['files']:
        print("\n✓ Plot file created successfully!")

    tool.clear_workspace()


def test_scenario_3_independent_task():
    """Test: Independent task - should create new file"""
    print("\n" + "="*80)
    print("TEST 3: Independent task (should create new file)")
    print("="*80)

    tool = PythonCoderTool(session_id="test_session_3")
    tool.clear_workspace()

    # First execution
    code1 = """
import pandas as pd
df = pd.DataFrame({'x': [1, 2, 3]})
print(df)
"""

    print("\n--- First execution ---")
    result1 = tool.execute(code1)
    print(f"Files: {list(result1['files'].keys())}")

    # Second execution: Completely different task
    code2 = """
import math
result = math.factorial(10)
print(f"Factorial of 10 is: {result}")
"""

    print("\n--- Second execution (different task) ---")
    result2 = tool.execute(code2)
    print(f"Success: {result2['success']}")
    print(f"Files: {list(result2['files'].keys())}")
    print(f"Output:\n{result2['stdout']}")

    # Should have 2 files
    if len(result2['files']) >= 2:
        print("\n✓ Created separate file for independent task!")

    tool.clear_workspace()


if __name__ == "__main__":
    print("Testing Python Coder Smart Edit Functionality")
    print("="*80)

    try:
        test_scenario_1_no_existing_files()
        test_scenario_2_edit_existing()
        test_scenario_3_independent_task()

        print("\n" + "="*80)
        print("ALL TESTS COMPLETED")
        print("="*80)

    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
