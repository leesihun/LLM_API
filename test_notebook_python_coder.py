"""
Test only the Python Code Generation cells from API_examples.ipynb
Cells 14-18 (Python Code Generation examples)
"""
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import sys

def test_python_coder_cells():
    """Execute only the Python code generation cells"""
    print("="*70)
    print("Testing Python Code Generation Cells from API_examples.ipynb")
    print("="*70)

    # Load the notebook
    with open("API_examples.ipynb", 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    print(f"\nTotal cells in notebook: {len(nb.cells)}")

    # Find the Python code generation cells (14-18)
    # These should be the last markdown + code cell pairs
    python_cells = []
    cell_descriptions = []

    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'markdown' and '## 14)' in cell.source:
            python_cells.append(i)
            cell_descriptions.append("Cell 14: Simple Calculation")
        elif cell.cell_type == 'markdown' and '## 15)' in cell.source:
            python_cells.append(i)
            cell_descriptions.append("Cell 15: Data Analysis")
        elif cell.cell_type == 'markdown' and '## 16)' in cell.source:
            python_cells.append(i)
            cell_descriptions.append("Cell 16: Mathematical Computation")
        elif cell.cell_type == 'markdown' and '## 17)' in cell.source:
            python_cells.append(i)
            cell_descriptions.append("Cell 17: String Processing")
        elif cell.cell_type == 'markdown' and '## 18)' in cell.source:
            python_cells.append(i)
            cell_descriptions.append("Cell 18: Excel File Analysis")

    print(f"\nFound {len(python_cells)} Python code generation sections")

    # Create a minimal notebook with just the setup and Python coder cells
    test_nb = nbformat.v4.new_notebook()

    # Add setup cells (client definition and login)
    # Cell 0: Imports and client class
    setup_code = """
import httpx

BASE_URL = "http://127.0.0.1:8000"
MODEL = "gpt-oss:20b"

class LLMApiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.token = None

    def login(self, username: str, password: str):
        r = httpx.post(f"{self.base_url}/api/auth/login", json={
            "username": username, "password": password
        }, timeout=10.0)
        r.raise_for_status()
        data = r.json()
        self.token = data["access_token"]
        return data

    def chat_new(self, model: str, message: str, agent_type: str = "auto"):
        r = httpx.post(f"{self.base_url}/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "agent_type": agent_type
        }, headers={"Authorization": f"Bearer {self.token}"}, timeout=180.0)
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        session_id = data.get("x_session_id")
        return content, session_id

client = LLMApiClient(BASE_URL)
"""

    test_nb.cells.append(nbformat.v4.new_code_cell(setup_code))

    # Cell 1: Login
    login_code = """
# Login with existing guest account
username = "guest"
password = "guest_test1"
client.login(username, password)
print(f"Logged in as {username}")
"""
    test_nb.cells.append(nbformat.v4.new_code_cell(login_code))

    # Add Python code generation cells
    for cell_idx in python_cells:
        # Add markdown header
        test_nb.cells.append(nb.cells[cell_idx])
        # Add the code cell that follows
        if cell_idx + 1 < len(nb.cells) and nb.cells[cell_idx + 1].cell_type == 'code':
            test_nb.cells.append(nb.cells[cell_idx + 1])

    print(f"Created test notebook with {len(test_nb.cells)} cells")

    # Execute the test notebook
    print("\nExecuting notebook (this may take several minutes)...")
    print("="*70)

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    try:
        ep.preprocess(test_nb, {'metadata': {'path': '.'}})

        print("\n" + "="*70)
        print("EXECUTION COMPLETE")
        print("="*70)

        # Count results
        code_cells = [cell for cell in test_nb.cells if cell.cell_type == 'code']
        executed_cells = [cell for cell in code_cells if cell.get('outputs')]
        error_cells = [cell for cell in code_cells if any(
            output.get('output_type') == 'error' for output in cell.get('outputs', [])
        )]

        print(f"\nCode cells: {len(code_cells)}")
        print(f"Executed cells: {len(executed_cells)}")
        print(f"Cells with errors: {len(error_cells)}")

        # Show errors if any
        if error_cells:
            print("\n" + "="*70)
            print("ERRORS FOUND")
            print("="*70)
            for i, cell in enumerate(test_nb.cells):
                if cell.cell_type == 'code':
                    for output in cell.get('outputs', []):
                        if output.get('output_type') == 'error':
                            print(f"\nCell {i}:")
                            print(f"Source: {cell.source[:100]}...")
                            print(f"Error: {output.get('ename', 'Unknown')}")
                            print(f"Message: {output.get('evalue', '')[:200]}")

        # Save result
        output_path = "API_examples_python_coder_test.ipynb"
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(test_nb, f)
        print(f"\nSaved test results to: {output_path}")

        if len(error_cells) == 0:
            print("\n[SUCCESS] All Python code generation cells executed without errors!")
            return True
        else:
            print(f"\n[WARNING] {len(error_cells)} cell(s) had errors")
            return False

    except Exception as e:
        print(f"\n[ERROR] Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_python_coder_cells()
    sys.exit(0 if success else 1)
