#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test difficult math problems with the LLM API
"""

import requests
import json
import time
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    import os
    os.system('chcp 65001 > nul 2>&1')

BASE_URL = "http://localhost:8000"

def test_difficult_math():
    """Test the API with challenging math problems"""

    print("\n" + "="*70)
    print("DIFFICULT MATH TEST - LLM API")
    print("="*70 + "\n")

    # Login first
    print("[1] Logging in...")
    login_response = requests.post(f"{BASE_URL}/api/auth/login",
                                   json={"username": "guest", "password": "guest_test1"})
    if login_response.status_code != 200:
        print(f"Login failed: {login_response.text}")
        return

    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print(f"[OK] Logged in successfully\n")

    # Test cases - from easy to difficult
    math_problems = [
        {
            "name": "Multi-step word problem (Average Speed)",
            "question": "A train travels 120 km in 1.5 hours, then slows down and travels 80 km in 2 hours. What is the average speed for the entire journey? Show your calculation step by step."
        },
        {
            "name": "Algebra (System of Equations)",
            "question": "Solve for x and y: 2x + 3y = 13 and 4x - y = 5. Show all steps."
        },
        {
            "name": "Percentage and Compound Interest",
            "question": "If you invest $1000 at 5% annual compound interest, how much will you have after 3 years? Show the formula and calculation."
        },
        {
            "name": "Geometry (Circle Area)",
            "question": "A circle has a diameter of 14 cm. Calculate its area. Use π ≈ 3.14159 and show your work."
        },
        {
            "name": "Complex Multi-step Problem",
            "question": "A store sells apples for $2 each and oranges for $3 each. Yesterday they sold 50 fruits total and made $130. How many apples and oranges did they sell? Solve step by step."
        }
    ]

    results = []

    for i, problem in enumerate(math_problems, 1):
        print(f"\n{'='*70}")
        print(f"[{i}] TEST: {problem['name']}")
        print(f"{'='*70}")
        print(f"\nQuestion: {problem['question']}\n")

        start_time = time.time()

        try:
            response = requests.post(
                f"{BASE_URL}/v1/chat/completions",
                headers=headers,
                json={
                    "model": "llama2",
                    "messages": [
                        {"role": "user", "content": problem['question']}
                    ]
                },
                timeout=90
            )

            elapsed = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                answer = result.get("message", {}).get("content", "No answer provided")

                print(f"Status: [SUCCESS] (took {elapsed:.1f}s)")
                print(f"\nLLM Response:")
                print("-" * 70)
                print(answer)
                print("-" * 70)

                results.append({
                    "problem": problem['name'],
                    "status": "PASS",
                    "time": elapsed,
                    "answer": answer
                })
            else:
                print(f"Status: [FAILED] (HTTP {response.status_code})")
                print(f"Error: {response.text}")
                results.append({
                    "problem": problem['name'],
                    "status": "FAIL",
                    "time": elapsed,
                    "error": response.text
                })

        except requests.exceptions.Timeout:
            print(f"Status: [TIMEOUT] (>90s)")
            results.append({
                "problem": problem['name'],
                "status": "TIMEOUT",
                "time": 90
            })
        except Exception as e:
            print(f"Status: [ERROR] - {str(e)}")
            results.append({
                "problem": problem['name'],
                "status": "ERROR",
                "error": str(e)
            })

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    passed = sum(1 for r in results if r["status"] == "PASS")
    total = len(results)

    for r in results:
        status_symbol = "[OK]" if r["status"] == "PASS" else "[X]"
        time_str = f"({r.get('time', 0):.1f}s)" if "time" in r else ""
        print(f"{status_symbol} {r['problem']:<50} {r['status']:<10} {time_str}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] ALL MATH TESTS PASSED!")
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed or timed out")

if __name__ == "__main__":
    test_difficult_math()
