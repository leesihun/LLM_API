"""
Simple test script to verify search engine dependencies
Run this before using the Jupyter notebook
"""

import sys

def check_dependencies():
    """Check if required packages are installed"""
    print("Checking search engine dependencies...\n")

    required_packages = {
        'langchain_community': 'langchain-community',
        'duckduckgo_search': 'duckduckgo-search',
        'tavily': 'tavily-python (optional)',
    }

    all_good = True

    for package, install_name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {install_name} is installed")
        except ImportError:
            print(f"✗ {install_name} is NOT installed")
            if 'optional' not in install_name:
                all_good = False

    print("\n" + "="*60)

    if all_good:
        print("✓ All required dependencies are installed!")
        print("\nYou can now run: jupyter notebook test_search_engines.ipynb")
    else:
        print("⚠ Missing dependencies detected")
        print("\nInstall missing packages with:")
        print("  pip install langchain-community duckduckgo-search tavily-python")

    print("="*60)

    return all_good


def test_duckduckgo():
    """Quick test of DuckDuckGo search"""
    print("\n\nTesting DuckDuckGo search (no API key required)...\n")

    try:
        from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

        search = DuckDuckGoSearchAPIWrapper(max_results=3)
        results = search.results("Python programming", 3)

        print(f"✓ DuckDuckGo search successful! Found {len(results)} results:\n")

        for i, result in enumerate(results, 1):
            print(f"{i}. {result.get('title', 'No title')}")
            print(f"   URL: {result.get('link', 'No URL')}")
            print(f"   {result.get('snippet', 'No snippet')[:100]}...\n")

        return True

    except Exception as e:
        print(f"✗ DuckDuckGo search failed: {e}")
        return False


if __name__ == "__main__":
    deps_ok = check_dependencies()

    if deps_ok:
        test_duckduckgo()
    else:
        print("\n⚠ Please install missing dependencies first")
        sys.exit(1)
