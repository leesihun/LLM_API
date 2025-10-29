"""
Execute API_examples.ipynb and generate a test report
"""
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import sys

def run_notebook(notebook_path):
    """Execute a Jupyter notebook and return results"""
    print(f"Loading notebook: {notebook_path}")

    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Configure the executor
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    print(f"Executing notebook...")
    print(f"Total cells: {len(nb.cells)}")
    print("="*70)

    try:
        # Execute the notebook
        ep.preprocess(nb, {'metadata': {'path': '.'}})

        print("\n" + "="*70)
        print("NOTEBOOK EXECUTION COMPLETE")
        print("="*70)

        # Count results
        total_cells = len(nb.cells)
        code_cells = sum(1 for cell in nb.cells if cell.cell_type == 'code')
        executed_cells = sum(1 for cell in nb.cells if cell.cell_type == 'code' and cell.get('outputs'))
        error_cells = sum(1 for cell in nb.cells if cell.cell_type == 'code' and any(
            output.get('output_type') == 'error' for output in cell.get('outputs', [])
        ))

        print(f"\nTotal cells: {total_cells}")
        print(f"Code cells: {code_cells}")
        print(f"Executed cells: {executed_cells}")
        print(f"Cells with errors: {error_cells}")

        # Show errors if any
        if error_cells > 0:
            print("\n" + "="*70)
            print("ERRORS FOUND")
            print("="*70)
            for i, cell in enumerate(nb.cells):
                if cell.cell_type == 'code':
                    for output in cell.get('outputs', []):
                        if output.get('output_type') == 'error':
                            print(f"\nCell {i}:")
                            print(f"Error: {output.get('ename', 'Unknown')}")
                            print(f"Message: {output.get('evalue', '')}")

        # Save executed notebook
        output_path = notebook_path.replace('.ipynb', '_executed.ipynb')
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"\nSaved executed notebook to: {output_path}")

        if error_cells == 0:
            print("\n[SUCCESS] All cells executed without errors!")
            return True
        else:
            print(f"\n[WARNING] {error_cells} cell(s) had errors")
            return False

    except Exception as e:
        print(f"\n[ERROR] Notebook execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    notebook_path = "API_examples.ipynb"
    success = run_notebook(notebook_path)
    sys.exit(0 if success else 1)
