"""
RAG Module Test Script
Tests the RAG tool directly and via API endpoints
"""
import sys
import os
import time
import json
import requests
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# Configuration
# =============================================================================

API_BASE_URL = "http://localhost:10007"
TEST_USERNAME = "test_rag_user"
TEST_PASSWORD = "test_password_123"
TEST_COLLECTION = "test_ml_docs"

# Test data paths
TEST_DATA_DIR = Path(__file__).parent / "test_data"
PDF_FILE = TEST_DATA_DIR / "ml_introduction.pdf"
HANDBOOK_FILE = TEST_DATA_DIR / "company_handbook.pdf"


# =============================================================================
# Direct RAGTool Tests (No server required)
# =============================================================================

def test_rag_tool_direct():
    """Test RAGTool directly without API server"""
    print("\n" + "=" * 80)
    print("DIRECT RAG TOOL TEST")
    print("=" * 80)

    from tools.rag import RAGTool

    username = "direct_test_user"
    collection_name = "direct_test_collection"

    # Initialize tool
    print("\n[1] Initializing RAGTool...")
    tool = RAGTool(username=username)
    print(f"    User docs dir: {tool.user_docs_dir}")
    print(f"    User index dir: {tool.user_index_dir}")
    print("    [OK] RAGTool initialized")

    # Create collection
    print("\n[2] Creating collection...")
    result = tool.create_collection(collection_name)
    if result["success"]:
        print(f"    [OK] Collection '{collection_name}' created")
    else:
        if "already exists" in result.get("error", ""):
            print(f"    [INFO] Collection already exists, continuing...")
        else:
            print(f"    [FAIL] {result.get('error')}")
            return False

    # Upload text document
    print("\n[3] Uploading text document...")
    text_content = """
    Machine learning is a method of data analysis that automates analytical model building.
    It is a branch of artificial intelligence based on the idea that systems can learn from data,
    identify patterns and make decisions with minimal human intervention.

    Deep learning is a subset of machine learning in artificial intelligence that has networks
    capable of learning unsupervised from data that is unstructured or unlabeled.

    Neural networks are computing systems inspired by biological neural networks that constitute
    animal brains. These systems learn to perform tasks by considering examples.
    """
    result = tool.upload_document(
        collection_name=collection_name,
        document_path="test_ml_basics.txt",
        document_content=text_content
    )
    if result["success"]:
        print(f"    [OK] Text document uploaded")
        print(f"        Chunks created: {result['chunks_created']}")
        print(f"        Total chunks: {result['total_chunks']}")
    else:
        print(f"    [FAIL] {result.get('error')}")
        return False

    # Upload PDF if available
    if PDF_FILE.exists():
        print("\n[4] Uploading PDF document...")
        result = tool.upload_document(
            collection_name=collection_name,
            document_path=str(PDF_FILE)
        )
        if result["success"]:
            print(f"    [OK] PDF document uploaded")
            print(f"        Chunks created: {result['chunks_created']}")
            print(f"        Total chunks: {result['total_chunks']}")
        else:
            print(f"    [FAIL] {result.get('error')}")
    else:
        print("\n[4] Skipping PDF test (file not found)")

    # List collections
    print("\n[5] Listing collections...")
    result = tool.list_collections()
    if result["success"]:
        print(f"    [OK] Found {len(result['collections'])} collection(s)")
        for coll in result["collections"]:
            print(f"        - {coll['name']}: {coll['documents']} docs, {coll['chunks']} chunks")
    else:
        print(f"    [FAIL] {result.get('error')}")

    # List documents in collection
    print("\n[6] Listing documents in collection...")
    result = tool.list_documents(collection_name)
    if result["success"]:
        print(f"    [OK] Found {result['total_documents']} document(s)")
        for doc in result["documents"]:
            print(f"        - {doc['name']}: {doc['chunks']} chunks")
    else:
        print(f"    [FAIL] {result.get('error')}")

    # Test retrieval
    print("\n[7] Testing retrieval...")
    queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "What is deep learning?"
    ]

    for query in queries:
        print(f"\n    Query: '{query}'")
        result = tool.retrieve(
            collection_name=collection_name,
            query=query,
            max_results=3
        )
        if result["success"]:
            print(f"    [OK] Retrieved {result['num_results']} result(s)")
            for i, doc in enumerate(result["documents"][:2], 1):
                preview = doc["chunk"][:100].replace("\n", " ")
                print(f"        {i}. [{doc['document']}] score={doc['score']:.3f}")
                print(f"           {preview}...")
        else:
            print(f"    [FAIL] {result.get('error')}")

    # Cleanup - delete collection
    print("\n[8] Cleaning up - deleting test collection...")
    result = tool.delete_collection(collection_name)
    if result["success"]:
        print(f"    [OK] Collection deleted")
    else:
        print(f"    [FAIL] {result.get('error')}")

    print("\n" + "-" * 80)
    print("DIRECT RAG TOOL TEST COMPLETED SUCCESSFULLY")
    print("-" * 80)
    return True


# =============================================================================
# API Tests (Server required)
# =============================================================================

class APITester:
    """Helper class for API testing"""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.token = None

    def signup(self, username: str, password: str) -> bool:
        """Create test user"""
        try:
            resp = requests.post(
                f"{self.base_url}/api/auth/signup",
                json={"username": username, "password": password}
            )
            if resp.status_code == 200:
                return True
            elif resp.status_code == 400 and "already exists" in resp.text.lower():
                return True
            else:
                print(f"Signup failed: {resp.status_code} - {resp.text}")
                return False
        except Exception as e:
            print(f"Signup error: {e}")
            return False

    def login(self, username: str, password: str) -> bool:
        """Login and get token"""
        try:
            resp = requests.post(
                f"{self.base_url}/api/auth/login",
                json={"username": username, "password": password}
            )
            if resp.status_code == 200:
                data = resp.json()
                self.token = data.get("access_token")
                return True
            else:
                print(f"Login failed: {resp.status_code} - {resp.text}")
                return False
        except Exception as e:
            print(f"Login error: {e}")
            return False

    def get_headers(self) -> dict:
        """Get auth headers"""
        if self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}


def test_rag_api():
    """Test RAG API endpoints"""
    print("\n" + "=" * 80)
    print("RAG API TEST")
    print("=" * 80)

    # Check if server is running
    print("\n[0] Checking server availability...")
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if resp.status_code != 200:
            print(f"    [FAIL] Server not healthy: {resp.status_code}")
            return False
        print("    [OK] Server is running")
    except requests.exceptions.ConnectionError:
        print(f"    [SKIP] Server not running at {API_BASE_URL}")
        print("    Start the server with: python run_backend.py")
        return None  # Return None to indicate skip
    except Exception as e:
        print(f"    [FAIL] Server check failed: {e}")
        return False

    # Initialize tester
    tester = APITester(API_BASE_URL)

    # Signup and login
    print("\n[1] Setting up test user...")
    if not tester.signup(TEST_USERNAME, TEST_PASSWORD):
        print("    [FAIL] Could not create test user")
        return False
    if not tester.login(TEST_USERNAME, TEST_PASSWORD):
        print("    [FAIL] Could not login")
        return False
    print("    [OK] User authenticated")

    headers = tester.get_headers()

    # Create collection
    print("\n[2] Creating collection via API...")
    resp = requests.post(
        f"{API_BASE_URL}/api/tools/rag/collections",
        headers=headers,
        json={"collection_name": TEST_COLLECTION}
    )
    if resp.status_code == 200:
        data = resp.json()
        if data.get("success"):
            print(f"    [OK] Collection '{TEST_COLLECTION}' created")
        elif "already exists" in data.get("answer", ""):
            print("    [INFO] Collection already exists")
        else:
            print(f"    [FAIL] {data}")
    else:
        print(f"    [FAIL] {resp.status_code} - {resp.text}")

    # Upload text file
    print("\n[3] Uploading text file via API...")
    text_content = """
    Retrieval-Augmented Generation (RAG) is an AI framework that combines information retrieval
    with text generation. It enhances language models by allowing them to access external knowledge
    bases, making responses more accurate and up-to-date.

    RAG systems typically consist of two main components:
    1. A retriever that finds relevant documents from a knowledge base
    2. A generator that produces responses based on the retrieved information

    FAISS (Facebook AI Similarity Search) is commonly used for efficient similarity search
    in RAG systems. It supports various index types for different accuracy/speed trade-offs.
    """
    files = {
        "file": ("rag_overview.txt", text_content, "text/plain")
    }
    data = {"collection_name": TEST_COLLECTION}
    resp = requests.post(
        f"{API_BASE_URL}/api/tools/rag/upload",
        headers=headers,
        files=files,
        data=data
    )
    if resp.status_code == 200:
        result = resp.json()
        if result.get("success"):
            print(f"    [OK] Text file uploaded")
            print(f"        Chunks: {result.get('chunks_created')}")
        else:
            print(f"    [FAIL] {result}")
    else:
        print(f"    [FAIL] {resp.status_code} - {resp.text}")

    # Upload PDF file
    if PDF_FILE.exists():
        print("\n[4] Uploading PDF file via API...")
        with open(PDF_FILE, "rb") as f:
            files = {
                "file": (PDF_FILE.name, f, "application/pdf")
            }
            data = {"collection_name": TEST_COLLECTION}
            resp = requests.post(
                f"{API_BASE_URL}/api/tools/rag/upload",
                headers=headers,
                files=files,
                data=data
            )
        if resp.status_code == 200:
            result = resp.json()
            if result.get("success"):
                print(f"    [OK] PDF file uploaded")
                print(f"        Document: {result.get('document_name')}")
                print(f"        Chunks: {result.get('chunks_created')}")
            else:
                print(f"    [FAIL] {result}")
        else:
            print(f"    [FAIL] {resp.status_code} - {resp.text}")
    else:
        print("\n[4] Skipping PDF upload (file not found)")

    # List collections
    print("\n[5] Listing collections via API...")
    resp = requests.get(
        f"{API_BASE_URL}/api/tools/rag/collections",
        headers=headers
    )
    if resp.status_code == 200:
        result = resp.json()
        if result.get("success"):
            print(f"    [OK] Found {len(result['collections'])} collection(s)")
            for coll in result["collections"]:
                print(f"        - {coll['name']}: {coll['documents']} docs")
        else:
            print(f"    [FAIL] {result}")
    else:
        print(f"    [FAIL] {resp.status_code} - {resp.text}")

    # List documents
    print("\n[6] Listing documents via API...")
    resp = requests.get(
        f"{API_BASE_URL}/api/tools/rag/collections/{TEST_COLLECTION}/documents",
        headers=headers
    )
    if resp.status_code == 200:
        result = resp.json()
        if result.get("success"):
            print(f"    [OK] Found {result['total_documents']} document(s)")
            for doc in result["documents"]:
                print(f"        - {doc['name']}: {doc['chunks']} chunks")
        else:
            print(f"    [FAIL] {result}")
    else:
        print(f"    [FAIL] {resp.status_code} - {resp.text}")

    # Query RAG (Note: This requires LLM backend to be running)
    print("\n[7] Querying RAG via API...")
    print("    (Note: This requires LLM backend to be running)")
    resp = requests.post(
        f"{API_BASE_URL}/api/tools/rag/query",
        headers=headers,
        json={
            "query": "What is RAG and how does it work?",
            "collection_name": TEST_COLLECTION,
            "max_results": 3
        }
    )
    if resp.status_code == 200:
        result = resp.json()
        if result.get("success"):
            print(f"    [OK] Query successful")
            print(f"        Optimized query: {result['data'].get('optimized_query', 'N/A')}")
            print(f"        Results: {result['data'].get('num_results', 0)}")
            answer = result.get("answer", "")[:200]
            print(f"        Answer preview: {answer}...")
        else:
            print(f"    [INFO] Query returned error (LLM may not be running)")
            print(f"           Error: {result.get('error', 'Unknown')}")
    else:
        print(f"    [INFO] Query failed (LLM may not be running)")
        print(f"           Status: {resp.status_code}")

    # Cleanup - delete collection
    print("\n[8] Cleaning up - deleting test collection...")
    resp = requests.delete(
        f"{API_BASE_URL}/api/tools/rag/collections/{TEST_COLLECTION}",
        headers=headers
    )
    if resp.status_code == 200:
        result = resp.json()
        if result.get("success"):
            print(f"    [OK] Collection deleted")
        else:
            print(f"    [FAIL] {result}")
    else:
        print(f"    [FAIL] {resp.status_code} - {resp.text}")

    print("\n" + "-" * 80)
    print("RAG API TEST COMPLETED")
    print("-" * 80)
    return True


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all RAG tests"""
    print("=" * 80)
    print("RAG MODULE TEST SUITE")
    print("=" * 80)
    print(f"Test data directory: {TEST_DATA_DIR}")
    print(f"PDF file exists: {PDF_FILE.exists()}")

    # Create test data if needed
    if not PDF_FILE.exists():
        print("\n[INFO] Creating test PDFs...")
        from tests.create_sample_pdf import create_sample_pdf, create_company_handbook_pdf
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        create_sample_pdf(str(PDF_FILE))
        create_company_handbook_pdf(str(HANDBOOK_FILE))

    results = {}

    # Run direct tool test
    try:
        results["direct"] = test_rag_tool_direct()
    except Exception as e:
        print(f"\n[ERROR] Direct test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results["direct"] = False

    # Run API test
    try:
        results["api"] = test_rag_api()
    except Exception as e:
        print(f"\n[ERROR] API test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results["api"] = False

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, result in results.items():
        if result is True:
            status = "PASSED"
        elif result is False:
            status = "FAILED"
        else:
            status = "SKIPPED"
        print(f"  {test_name.upper()}: {status}")

    all_passed = all(r is True or r is None for r in results.values())
    if all_passed:
        print("\nAll tests passed or skipped!")
        return 0
    else:
        print("\nSome tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
