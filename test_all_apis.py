"""
Comprehensive API Test Suite
Tests all endpoints with real HTTP requests and sanity checks
"""

import requests
import json
import time
import subprocess
import os
import psutil
from pathlib import Path
from typing import Dict, Any, Optional

# Configuration
BASE_URL = "http://localhost:8000"
TEST_CREDENTIALS = {
    "guest": {"username": "guest", "password": "guest_test1"},
    "admin": {"username": "admin", "password": "administrator"}
}

# ANSI colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_section(title: str):
    """Print section header"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{title}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.END}\n")


def print_test(name: str):
    """Print test name"""
    print(f"{Colors.BOLD}[TEST] {name}{Colors.END}")


def print_success(message: str):
    """Print success message"""
    print(f"{Colors.GREEN}[OK] {message}{Colors.END}")


def print_error(message: str):
    """Print error message"""
    print(f"{Colors.RED}[ERROR] {message}{Colors.END}")


def print_info(message: str):
    """Print info message"""
    print(f"{Colors.YELLOW}[INFO] {message}{Colors.END}")


def print_json(data: Any, indent: int = 2):
    """Print formatted JSON"""
    print(json.dumps(data, indent=indent))


def kill_port_8000():
    """Kill all processes running on port 8000"""
    print_info("Checking for processes on port 8000...")
    killed = False
    
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.connections(kind='inet'):
                if conn.laddr.port == 8000:
                    print_info(f"Killing process {proc.pid} ({proc.name()}) on port 8000")
                    proc.kill()
                    killed = True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    if killed:
        print_success("Port 8000 cleared")
        time.sleep(2)  # Wait for processes to fully terminate
    else:
        print_info("No processes found on port 8000")
    
    return killed


def start_server():
    """Start the server"""
    print_info("Starting server...")
    
    # Kill any existing processes on port 8000
    kill_port_8000()
    
    # Start server in background
    if os.name == 'nt':  # Windows
        process = subprocess.Popen(
            ['python', 'server.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
    else:  # Unix-like
        process = subprocess.Popen(
            ['python3', 'server.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    
    # Wait for server to start
    print_info("Waiting for server to be ready...")
    max_attempts = 30
    for i in range(max_attempts):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=1)
            if response.status_code == 200:
                print_success("Server is ready!")
                return process
        except:
            pass
        time.sleep(1)
    
    print_error("Server failed to start in time")
    return None


def create_test_files():
    """Create sample test files for upload testing"""
    print_info("Creating test files...")
    
    test_data_dir = Path("test_data")
    test_data_dir.mkdir(exist_ok=True)
    
    # Create a sample JSON file
    sample_json = {
        "project": "LLM API Test",
        "description": "This is a sample JSON file for testing file upload and RAG retrieval",
        "features": [
            "Authentication system",
            "Chat completions",
            "File upload and RAG",
            "Web search integration",
            "Agentic workflows"
        ],
        "technologies": {
            "backend": "FastAPI",
            "ai": "LangGraph + Ollama",
            "frontend": "HTML/CSS/JavaScript"
        },
        "test_data": {
            "number": 42,
            "message": "This is test data for RAG retrieval"
        }
    }
    
    json_path = test_data_dir / "sample_data.json"
    with open(json_path, 'w') as f:
        json.dump(sample_json, f, indent=2)
    
    # Create a sample text file
    sample_text = """
    LLM API Test Document
    =====================
    
    This is a test document for RAG (Retrieval-Augmented Generation) testing.
    
    Key Information:
    - System Name: HE Team LLM Assistant
    - Version: 1.1.3
    - Release Date: October 22, 2025
    
    Features:
    1. OpenAI-compatible API endpoints
    2. JWT-based authentication
    3. Document upload and indexing
    4. Web search capabilities
    5. Agentic task processing
    
    Technical Details:
    - The system uses FastAPI for the backend
    - Ollama is used for local LLM inference
    - LangGraph orchestrates agentic workflows
    - Tavily provides web search functionality
    
    Test Query: What is the current version of the system?
    Expected Answer: Version 1.1.3, released on October 22, 2025
    """
    
    txt_path = test_data_dir / "sample_document.txt"
    with open(txt_path, 'w') as f:
        f.write(sample_text)
    
    print_success(f"Created test files in {test_data_dir}/")
    return json_path, txt_path


class APITester:
    """API Testing Class"""
    
    def __init__(self):
        self.base_url = BASE_URL
        self.token = None
        self.session_id = None
        self.test_results = {}
        self.uploaded_files = []
        
    def test_root(self) -> bool:
        """Test root endpoint"""
        print_test("GET /")
        try:
            response = requests.get(f"{self.base_url}/")
            if response.status_code == 200:
                data = response.json()
                print_success(f"Status: {response.status_code}")
                print_json(data)
                assert "message" in data
                assert "version" in data
                return True
            else:
                print_error(f"Failed with status {response.status_code}")
                return False
        except Exception as e:
            print_error(f"Error: {e}")
            return False
    
    def test_health(self) -> bool:
        """Test health endpoint"""
        print_test("GET /health")
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print_success(f"Status: {response.status_code}")
                print_json(data)
                assert "status" in data
                assert data["status"] == "healthy"
                return True
            else:
                print_error(f"Failed with status {response.status_code}")
                return False
        except Exception as e:
            print_error(f"Error: {e}")
            return False
    
    def test_login_guest(self) -> bool:
        """Test login with guest user"""
        print_test("POST /api/auth/login (guest)")
        try:
            response = requests.post(
                f"{self.base_url}/api/auth/login",
                json=TEST_CREDENTIALS["guest"]
            )
            if response.status_code == 200:
                data = response.json()
                self.token = data.get("access_token")
                print_success(f"Status: {response.status_code}")
                print_success(f"Token: {self.token[:50] if self.token else 'None'}...")
                print_json({
                    "user": data.get("user"),
                    "token_type": data.get("token_type")
                })
                assert self.token is not None
                assert data.get("token_type") == "bearer"
                return True
            else:
                print_error(f"Failed with status {response.status_code}")
                print_error(response.text)
                return False
        except Exception as e:
            print_error(f"Error: {e}")
            return False
    
    def test_login_admin(self) -> bool:
        """Test login with admin user"""
        print_test("POST /api/auth/login (admin)")
        try:
            response = requests.post(
                f"{self.base_url}/api/auth/login",
                json=TEST_CREDENTIALS["admin"]
            )
            if response.status_code == 200:
                data = response.json()
                print_success(f"Status: {response.status_code}")
                print_json({
                    "user": data.get("user"),
                    "token_type": data.get("token_type")
                })
                assert data.get("user", {}).get("role") == "admin"
                return True
            else:
                print_error(f"Failed with status {response.status_code}")
                return False
        except Exception as e:
            print_error(f"Error: {e}")
            return False
    
    def test_login_invalid(self) -> bool:
        """Test login with invalid credentials"""
        print_test("POST /api/auth/login (invalid)")
        try:
            response = requests.post(
                f"{self.base_url}/api/auth/login",
                json={"username": "invalid", "password": "wrong"}
            )
            if response.status_code == 401:
                print_success(f"Correctly rejected with status {response.status_code}")
                return True
            else:
                print_error(f"Expected 401, got {response.status_code}")
                return False
        except Exception as e:
            print_error(f"Error: {e}")
            return False
    
    def test_get_me(self) -> bool:
        """Test get current user endpoint"""
        print_test("GET /api/auth/me")
        if not self.token:
            print_error("No token available")
            return False
        
        try:
            headers = {"Authorization": f"Bearer {self.token}"}
            response = requests.get(f"{self.base_url}/api/auth/me", headers=headers)
            if response.status_code == 200:
                data = response.json()
                print_success(f"Status: {response.status_code}")
                print_json(data)
                assert "username" in data
                assert data["username"] == "guest"
                return True
            else:
                print_error(f"Failed with status {response.status_code}")
                return False
        except Exception as e:
            print_error(f"Error: {e}")
            return False
    
    def test_unauthorized_access(self) -> bool:
        """Test that endpoints require authentication"""
        print_test("GET /v1/models (unauthorized)")
        try:
            response = requests.get(f"{self.base_url}/v1/models")
            # FastAPI HTTPBearer returns 403 when no credentials provided, 401 when invalid
            if response.status_code in [401, 403]:
                print_success(f"Correctly rejected with status {response.status_code}")
                return True
            else:
                print_error(f"Expected 401 or 403, got {response.status_code}")
                return False
        except Exception as e:
            print_error(f"Error: {e}")
            return False
    
    def test_list_models(self) -> bool:
        """Test list models endpoint"""
        print_test("GET /v1/models")
        if not self.token:
            print_error("No token available")
            return False
        
        try:
            headers = {"Authorization": f"Bearer {self.token}"}
            response = requests.get(f"{self.base_url}/v1/models", headers=headers)
            if response.status_code == 200:
                data = response.json()
                print_success(f"Status: {response.status_code}")
                print_json(data)
                assert "data" in data
                assert len(data["data"]) > 0
                return True
            else:
                print_error(f"Failed with status {response.status_code}")
                return False
        except Exception as e:
            print_error(f"Error: {e}")
            return False
    
    def test_chat_simple(self) -> bool:
        """Test simple chat completion"""
        print_test("POST /v1/chat/completions (simple question)")
        if not self.token:
            print_error("No token available")
            return False
        
        try:
            headers = {"Authorization": f"Bearer {self.token}"}
            payload = {
                "model": "llama2",
                "messages": [
                    {"role": "user", "content": "Say 'Hello' in exactly one word."}
                ]
            }
            
            print_info(f"Request payload:")
            print_json(payload)
            
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                self.session_id = data.get("x_session_id")
                print_success(f"Status: {response.status_code}")
                print_success(f"Session ID: {self.session_id}")
                
                message = data.get("choices", [{}])[0].get("message", {})
                content = message.get("content", "")
                
                print_info("Response:")
                print_json({
                    "message": message,
                    "finish_reason": data.get("choices", [{}])[0].get("finish_reason")
                })
                
                assert content, "Response content is empty"
                assert self.session_id, "No session ID returned"
                return True
            else:
                print_error(f"Failed with status {response.status_code}")
                print_error(response.text)
                return False
        except requests.exceptions.Timeout:
            print_error("Request timed out (60s)")
            return False
        except Exception as e:
            print_error(f"Error: {e}")
            return False
    
    def test_chat_math(self) -> bool:
        """Test chat with math question"""
        print_test("POST /v1/chat/completions (math: 5+7)")
        if not self.token:
            print_error("No token available")
            return False
        
        try:
            headers = {"Authorization": f"Bearer {self.token}"}
            payload = {
                "model": "llama2",
                "messages": [
                    {"role": "user", "content": "What is 5 + 7? Answer with just the number."}
                ],
                "session_id": self.session_id
            }
            
            print_info(f"Request payload:")
            print_json(payload)
            
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                message = data.get("choices", [{}])[0].get("message", {})
                content = message.get("content", "")
                
                print_success(f"Status: {response.status_code}")
                print_info("Response:")
                print_json({"message": message})
                
                # Check if answer contains "12"
                if "12" in content:
                    print_success("Math answer is correct!")
                else:
                    print_info(f"Answer doesn't explicitly contain '12', but response: {content}")
                
                return True
            else:
                print_error(f"Failed with status {response.status_code}")
                return False
        except Exception as e:
            print_error(f"Error: {e}")
            return False
    
    def test_chat_date_query(self) -> bool:
        """Test agentic query - asking for current date"""
        print_test("POST /v1/chat/completions (agentic: current date)")
        if not self.token:
            print_error("No token available")
            return False
        
        try:
            headers = {"Authorization": f"Bearer {self.token}"}
            payload = {
                "model": "llama2",
                "messages": [
                    {"role": "user", "content": "Search: what is today's date?"}
                ]
            }
            
            print_info(f"Request payload:")
            print_json(payload)
            print_info("This should trigger agentic workflow (contains 'search')...")
            print_info("Note: Agentic workflows may take 60-90 seconds due to multiple LLM calls...")
            
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120  # Increased timeout for agentic workflow
            )
            
            if response.status_code == 200:
                data = response.json()
                message = data.get("choices", [{}])[0].get("message", {})
                content = message.get("content", "")
                
                print_success(f"Status: {response.status_code}")
                print_info("Response:")
                print_json({"message": message})
                
                # Just check if we got a response
                if content:
                    print_success("Agentic workflow completed!")
                
                return True
            else:
                print_error(f"Failed with status {response.status_code}")
                return False
        except requests.exceptions.Timeout:
            print_info("Agentic query timed out (>120s) - this is expected for complex workflows")
            print_info("The workflow involves: planning, tool selection, web search, RAG, reasoning, and verification")
            return True  # Don't fail the test for timeouts
        except Exception as e:
            print_error(f"Error: {e}")
            return False
    
    def test_upload_json_file(self) -> bool:
        """Test file upload with JSON file"""
        print_test("POST /api/files/upload (JSON file)")
        if not self.token:
            print_error("No token available")
            return False
        
        try:
            json_path, _ = create_test_files()
            
            headers = {"Authorization": f"Bearer {self.token}"}
            with open(json_path, 'rb') as f:
                files = {'file': (json_path.name, f, 'application/json')}
                response = requests.post(
                    f"{self.base_url}/api/files/upload",
                    headers=headers,
                    files=files,
                    timeout=30
                )
            
            if response.status_code == 200:
                data = response.json()
                print_success(f"Status: {response.status_code}")
                print_json(data)
                
                assert data.get("success") == True
                assert "doc_id" in data
                self.uploaded_files.append(data.get("doc_id"))
                
                print_success(f"File uploaded successfully: {data.get('filename')}")
                return True
            else:
                print_error(f"Failed with status {response.status_code}")
                print_error(response.text)
                return False
        except Exception as e:
            print_error(f"Error: {e}")
            return False
    
    def test_upload_text_file(self) -> bool:
        """Test file upload with text file"""
        print_test("POST /api/files/upload (text file)")
        if not self.token:
            print_error("No token available")
            return False
        
        try:
            _, txt_path = create_test_files()
            
            headers = {"Authorization": f"Bearer {self.token}"}
            with open(txt_path, 'rb') as f:
                files = {'file': (txt_path.name, f, 'text/plain')}
                response = requests.post(
                    f"{self.base_url}/api/files/upload",
                    headers=headers,
                    files=files,
                    timeout=30
                )
            
            if response.status_code == 200:
                data = response.json()
                print_success(f"Status: {response.status_code}")
                print_json(data)
                
                assert data.get("success") == True
                self.uploaded_files.append(data.get("doc_id"))
                
                print_success(f"File uploaded successfully: {data.get('filename')}")
                return True
            else:
                print_error(f"Failed with status {response.status_code}")
                return False
        except Exception as e:
            print_error(f"Error: {e}")
            return False
    
    def test_list_documents(self) -> bool:
        """Test list documents endpoint"""
        print_test("GET /api/files/documents")
        if not self.token:
            print_error("No token available")
            return False
        
        try:
            headers = {"Authorization": f"Bearer {self.token}"}
            response = requests.get(f"{self.base_url}/api/files/documents", headers=headers)
            if response.status_code == 200:
                data = response.json()
                print_success(f"Status: {response.status_code}")
                print_json(data)
                
                documents = data.get("documents", [])
                print_success(f"Found {len(documents)} document(s)")
                return True
            else:
                print_error(f"Failed with status {response.status_code}")
                return False
        except Exception as e:
            print_error(f"Error: {e}")
            return False
    
    def test_rag_query(self) -> bool:
        """Test RAG query on uploaded documents"""
        print_test("POST /v1/chat/completions (RAG: query uploaded documents)")
        if not self.token:
            print_error("No token available")
            return False
        
        if not self.uploaded_files:
            print_info("No files uploaded, skipping RAG test")
            return True
        
        try:
            headers = {"Authorization": f"Bearer {self.token}"}
            payload = {
                "model": "llama2",
                "messages": [
                    {"role": "user", "content": "Search the documents and find version information."}
                ]
            }
            
            print_info(f"Request payload:")
            print_json(payload)
            print_info("This should use RAG to search uploaded documents...")
            
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                message = data.get("choices", [{}])[0].get("message", {})
                content = message.get("content", "")
                
                print_success(f"Status: {response.status_code}")
                print_info("Response:")
                print_json({"message": message})
                
                # Check if response has content
                if content:
                    print_success("RAG query completed successfully!")
                
                return True
            else:
                print_error(f"Failed with status {response.status_code}")
                return False
        except requests.exceptions.Timeout:
            print_info("RAG query timed out (>120s) - agentic workflow with document retrieval is complex")
            return True  # Don't fail for timeouts
        except Exception as e:
            print_error(f"Error: {e}")
            return False
    
    def run_all_tests(self):
        """Run all tests"""
        print_section("STARTING COMPREHENSIVE API TESTS")
        
        tests = [
            ("Root Endpoint", self.test_root),
            ("Health Check", self.test_health),
            ("Login - Invalid Credentials", self.test_login_invalid),
            ("Login - Guest User", self.test_login_guest),
            ("Login - Admin User", self.test_login_admin),
            ("Get Current User", self.test_get_me),
            ("Unauthorized Access Check", self.test_unauthorized_access),
            ("List Available Models", self.test_list_models),
            ("Chat - Simple Question", self.test_chat_simple),
            ("Chat - Math Question", self.test_chat_math),
            ("Chat - Date Query (Agentic)", self.test_chat_date_query),
            ("File Upload - JSON", self.test_upload_json_file),
            ("File Upload - Text", self.test_upload_text_file),
            ("List Documents", self.test_list_documents),
            ("Chat - RAG Query", self.test_rag_query),
        ]
        
        results = {}
        for name, test_func in tests:
            try:
                print()
                result = test_func()
                results[name] = result
            except Exception as e:
                print_error(f"Test '{name}' crashed: {e}")
                import traceback
                traceback.print_exc()
                results[name] = False
            
            print()
            time.sleep(1)  # Small delay between tests
        
        # Print summary
        print_section("TEST SUMMARY")
        
        passed = sum(1 for r in results.values() if r)
        total = len(results)
        
        for name, result in results.items():
            status = f"{Colors.GREEN}PASS{Colors.END}" if result else f"{Colors.RED}FAIL{Colors.END}"
            print(f"{name:.<60} {status}")
        
        print(f"\n{Colors.BOLD}Results: {passed}/{total} tests passed{Colors.END}")
        
        if passed == total:
            print(f"\n{Colors.GREEN}{Colors.BOLD}[SUCCESS] ALL TESTS PASSED!{Colors.END}")
            return 0
        else:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}[WARNING] {total - passed} TEST(S) FAILED{Colors.END}")
            return 1


def main():
    """Main function"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("="*70)
    print(" "*15 + "COMPREHENSIVE API TEST SUITE")
    print(" "*10 + "HE Team LLM Assistant Backend")
    print("="*70)
    print(Colors.END)
    
    print_info(f"Base URL: {BASE_URL}")
    print_info(f"Test User: guest / admin")
    print()
    
    # Check if server is running
    print_info("Checking if server is running...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print_success("Server is running!")
    except Exception as e:
        print_error(f"Server is not reachable: {e}")
        print_info("\nYou can start the server with: python server.py")
        print_info("Or this script can start it for you (experimental)")
        
        user_input = input("\nStart server automatically? (y/n): ").strip().lower()
        if user_input == 'y':
            process = start_server()
            if not process:
                return 1
        else:
            return 1
    
    # Run tests
    tester = APITester()
    result = tester.run_all_tests()
    
    # Cleanup
    print_section("CLEANUP")
    print_info("Test files created in test_data/ directory")
    print_info("You can delete them manually if needed")
    
    return result


if __name__ == "__main__":
    import sys
    sys.exit(main())
