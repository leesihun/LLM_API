#!/usr/bin/env python3
"""
Test ReAct Agent Implementation
Demonstrates Thought-Action-Observation loop
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_react_agent():
    """Test ReAct agent with various query types"""

    print("\n" + "="*70)
    print("REACT AGENT TEST")
    print("="*70 + "\n")

    # Login
    print("[1] Logging in...")
    login_response = requests.post(f"{BASE_URL}/api/auth/login",
                                   json={"username": "guest", "password": "guest_test1"})
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("[OK] Logged in\n")

    # Test Cases
    test_cases = [
        {
            "name": "ReAct: Multi-step Query",
            "query": "First search for the capital of South Korea, then find information about its population",
            "agent_type": "react",
            "description": "Sequential query - should use ReAct for step-by-step reasoning"
        },
        {
            "name": "Plan-and-Execute: Parallel Query",
            "query": "Search for weather in Seoul AND analyze the max price in uploaded data",
            "agent_type": "plan_execute",
            "description": "Parallel operations - should use Plan-and-Execute"
        },
        {
            "name": "Auto-Select Agent",
            "query": "Find the maximum price in the data",
            "agent_type": "auto",
            "description": "Auto-select - system chooses best agent"
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}: {test['name']}")
        print(f"{'='*70}")
        print(f"\nDescription: {test['description']}")
        print(f"Query: {test['query']}")
        print(f"Agent Type: {test['agent_type']}\n")

        try:
            response = requests.post(
                f"{BASE_URL}/v1/chat/completions",
                headers=headers,
                json={
                    "model": "llama2",
                    "messages": [
                        {"role": "user", "content": test['query']}
                    ],
                    "agent_type": test['agent_type']
                },
                timeout=180  # 3 minutes for complex queries
            )

            if response.status_code == 200:
                result = response.json()
                answer = result.get("choices", [{}])[0].get("message", {}).get("content", "")

                print("[OK] Request completed")
                print(f"\nResponse:\n{answer}\n")
                print(f"Session ID: {result.get('x_session_id', 'N/A')}")
            else:
                print(f"[ERROR] Request failed with status {response.status_code}")
                print(f"Error: {response.text}")

        except requests.exceptions.Timeout:
            print("[TIMEOUT] Request took too long (>180s)")
        except Exception as e:
            print(f"[ERROR] {str(e)}")

    print(f"\n{'='*70}")
    print("COMPARISON: ReAct vs Plan-and-Execute")
    print(f"{'='*70}\n")

    print("ReAct Agent:")
    print("- Pattern: Thought → Action → Observation → Repeat")
    print("- Best for: Sequential reasoning, exploration, dynamic tool selection")
    print("- Use when: Steps depend on previous results")
    print("- Example: 'Find X, then based on X, do Y'")
    print()

    print("Plan-and-Execute Agent:")
    print("- Pattern: Plan Everything → Execute All → Verify")
    print("- Best for: Parallel operations, batch processing")
    print("- Use when: Multiple independent operations")
    print("- Example: 'Do X AND Y AND Z'")
    print()

    print(f"{'='*70}")
    print("CHECK SERVER LOGS")
    print(f"{'='*70}\n")
    print("Look for these log patterns in the server terminal:")
    print()
    print("ReAct Agent Logs:")
    print("  - [ReAct Agent] Starting for user: ...")
    print("  - [ReAct Agent] Iteration 1/5")
    print("  - [ReAct Agent] Thought: ...")
    print("  - [ReAct Agent] Action: web_search, Input: ...")
    print("  - [ReAct Agent] Observation: ...")
    print("  - [ReAct Agent] Iteration 2/5")
    print("  - ... (repeats until finish)")
    print()
    print("Plan-and-Execute Agent Logs:")
    print("  - [AGENT: Planning] Creating execution plan")
    print("  - [AGENT: Tool Selection] Selected tools: ...")
    print("  - [AGENT: Web Search] Performing web search")
    print("  - [AGENT: Reasoning] Generating final response")
    print()

if __name__ == "__main__":
    test_react_agent()
