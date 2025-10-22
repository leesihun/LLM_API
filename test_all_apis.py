"""
Comprehensive API Test Suite
Tests all endpoints with real HTTP requests
"""

import requests
import json
import time
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8000"
TEST_CREDENTIALS = {
    "guest": {"username": "guest", "password": "guest_test1"},
    "admin": {"username": "admin", "password": "admin_test1"}
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


class APITester:
    """API Testing Class"""
    
    def __init__(self):
        self.base_url = BASE_URL
        self.token = None
        self.session_id = None
        self.test_results = {}
        
    def test_root(self) -> bool:
        """Test root endpoint"""
        print_test("GET /")
        try:
            response = requests.get(f"{self.base_url}/")
            if response.status_code == 200:
                data = response.json()
                print_success(f"Status: {response.status_code}")
                print_json(data)
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
                print_success(f"Token: {self.token[:50]}...")
                print_json({
                    "user": data.get("user"),
                    "token_type": data.get("token_type")
                })
                return True
            else:
                print_error(f"Failed with status {response.status_code}")
                print_error(response.text)
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
                return True
            else:
                print_error(f"Failed with status {response.status_code}")
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
                return True
            else:
                print_error(f"Failed with status {response.status_code}")
                return False
        except Exception as e:
            print_error(f"Error: {e}")
            return False
    
    def test_chat_simple(self) -> bool:
        """Test simple chat completion"""
        print_test("POST /v1/chat/completions (simple)")
        if not self.token:
            print_error("No token available")
            return False
        
        try:
            headers = {"Authorization": f"Bearer {self.token}"}
            payload = {
                "model": "llama2",
                "messages": [
                    {"role": "user", "content": "Hello! Just say 'Hi' back."}
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
                print_info("Response:")
                print_json({
                    "message": data.get("choices", [{}])[0].get("message"),
                    "finish_reason": data.get("choices", [{}])[0].get("finish_reason")
                })
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
    
    def test_chat_with_session(self) -> bool:
        """Test chat completion with session"""
        print_test("POST /v1/chat/completions (with session)")
        if not self.token:
            print_error("No token available")
            return False
        
        if not self.session_id:
            print_info("No session ID, will create new session")
        
        try:
            headers = {"Authorization": f"Bearer {self.token}"}
            payload = {
                "model": "llama2",
                "messages": [
                    {"role": "user", "content": "What is 2+2?"}
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
                print_success(f"Status: {response.status_code}")
                print_info("Response:")
                print_json({
                    "message": data.get("choices", [{}])[0].get("message"),
                    "finish_reason": data.get("choices", [{}])[0].get("finish_reason")
                })
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
    
    def test_unauthorized_access(self) -> bool:
        """Test that endpoints require authentication"""
        print_test("GET /v1/models (unauthorized)")
        try:
            response = requests.get(f"{self.base_url}/v1/models")
            if response.status_code == 401:
                print_success(f"Correctly rejected with status {response.status_code}")
                return True
            else:
                print_error(f"Expected 401, got {response.status_code}")
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
                return True
            else:
                print_error(f"Failed with status {response.status_code}")
                return False
        except Exception as e:
            print_error(f"Error: {e}")
            return False
    
    def run_all_tests(self):
        """Run all tests"""
        print_section("STARTING API TESTS")
        
        tests = [
            ("Root Endpoint", self.test_root),
            ("Health Check", self.test_health),
            ("Login (Invalid)", self.test_login_invalid),
            ("Login (Guest)", self.test_login_guest),
            ("Get Current User", self.test_get_me),
            ("Unauthorized Access", self.test_unauthorized_access),
            ("List Models", self.test_list_models),
            ("List Documents", self.test_list_documents),
            ("Chat Completion (Simple)", self.test_chat_simple),
            ("Chat Completion (With Session)", self.test_chat_with_session),
        ]
        
        results = {}
        for name, test_func in tests:
            try:
                print()
                result = test_func()
                results[name] = result
            except Exception as e:
                print_error(f"Test '{name}' crashed: {e}")
                results[name] = False
            
            print()
            time.sleep(0.5)
        
        # Print summary
        print_section("TEST SUMMARY")
        
        passed = sum(1 for r in results.values() if r)
        total = len(results)
        
        for name, result in results.items():
            status = f"{Colors.GREEN}PASS{Colors.END}" if result else f"{Colors.RED}FAIL{Colors.END}"
            print(f"{name:.<50} {status}")
        
        print(f"\n{Colors.BOLD}Results: {passed}/{total} tests passed{Colors.END}")
        
        if passed == total:
            print(f"\n{Colors.GREEN}{Colors.BOLD}[SUCCESS] ALL TESTS PASSED!{Colors.END}")
            return 0
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}[FAILED] SOME TESTS FAILED{Colors.END}")
            return 1


def main():
    """Main function"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("="*70)
    print(" "*15 + "LIVE API TEST SUITE")
    print(" "*10 + "HE Team LLM Assistant Backend")
    print("="*70)
    print(Colors.END)
    
    print_info(f"Base URL: {BASE_URL}")
    print_info(f"Test User: guest")
    print()
    
    # Check if server is running
    print_info("Checking if server is running...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print_success("Server is running!")
    except Exception as e:
        print_error(f"Server is not reachable: {e}")
        print_error("Please start the server first: python server.py")
        return 1
    
    # Run tests
    tester = APITester()
    return tester.run_all_tests()


if __name__ == "__main__":
    import sys
    sys.exit(main())

