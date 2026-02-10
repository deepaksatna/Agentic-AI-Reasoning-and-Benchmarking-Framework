#!/usr/bin/env python3
"""
ARBM - API-based Benchmark Runner
Runs benchmarks against external APIs (Anthropic Claude)
"""

import os
import json
import time
import requests
from datetime import datetime

print("=" * 60)
print("  ARBM Benchmark - API Mode (Anthropic Claude)")
print("=" * 60)

# Configuration
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-3-5-sonnet-20241022"
API_URL = "https://api.anthropic.com/v1/messages"

# Benchmark tasks
TASKS = [
    {
        "id": "cot_math_001",
        "name": "Multi-step Math",
        "question": "A bakery sells cupcakes for $3 each. If they make 48 cupcakes and sell 75% of them, how much revenue do they generate?",
        "expected": "108"
    },
    {
        "id": "cot_logic_001", 
        "name": "Logical Deduction",
        "question": "If all programmers use computers, and some computer users are gamers, can we conclude that some programmers are gamers? Answer yes or no and explain.",
        "expected": "no"
    },
    {
        "id": "cot_reasoning_001",
        "name": "Multi-hop Reasoning",
        "question": "Alice is taller than Bob. Bob is taller than Charlie. David is shorter than Charlie. Who is the tallest?",
        "expected": "Alice"
    },
    {
        "id": "tool_planning_001",
        "name": "Task Planning",
        "question": "You need to deploy a machine learning model. List the steps in order: testing, training, data collection, deployment, monitoring.",
        "expected": "data collection, training, testing, deployment, monitoring"
    },
    {
        "id": "code_gen_001",
        "name": "Code Generation",
        "question": "Write a Python function to check if a number is prime. Just the function, no explanation.",
        "expected": "def"
    }
]

def call_anthropic(prompt):
    """Call Anthropic Claude API"""
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    data = {
        "model": MODEL,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    start_time = time.time()
    response = requests.post(API_URL, headers=headers, json=data, timeout=60)
    latency = time.time() - start_time
    
    if response.status_code == 200:
        result = response.json()
        content = result["content"][0]["text"]
        tokens_in = result["usage"]["input_tokens"]
        tokens_out = result["usage"]["output_tokens"]
        return {
            "success": True,
            "response": content,
            "latency": latency,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out
        }
    else:
        return {
            "success": False,
            "error": response.text,
            "latency": latency
        }

def main():
    if not ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY not set")
        return
    
    print(f"\nModel: {MODEL}")
    print(f"Tasks: {len(TASKS)}")
    print("-" * 60)
    
    results = []
    total_latency = 0
    total_tokens = 0
    successful = 0
    
    for task in TASKS:
        print(f"\n[{task[\"id\"]}] {task[\"name\"]}...")
        
        result = call_anthropic(task["question"])
        
        if result["success"]:
            successful += 1
            total_latency += result["latency"]
            total_tokens += result["tokens_in"] + result["tokens_out"]
            
            # Check if expected answer is in response
            correct = task["expected"].lower() in result["response"].lower()
            
            print(f"  ✓ Latency: {result[\"latency\"]:.2f}s")
            print(f"  ✓ Tokens: {result[\"tokens_in\"]} in / {result[\"tokens_out\"]} out")
            print(f"  ✓ Correct: {correct}")
            
            results.append({
                "task_id": task["id"],
                "task_name": task["name"],
                "success": True,
                "correct": correct,
                "latency": result["latency"],
                "tokens_in": result["tokens_in"],
                "tokens_out": result["tokens_out"],
                "response_preview": result["response"][:200]
            })
        else:
            print(f"  ✗ Error: {result[\"error\"][:100]}")
            results.append({
                "task_id": task["id"],
                "task_name": task["name"],
                "success": False,
                "error": result["error"][:200]
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Tasks Run: {len(TASKS)}")
    print(f"  Successful: {successful}")
    print(f"  Success Rate: {100*successful/len(TASKS):.1f}%")
    if successful > 0:
        print(f"  Avg Latency: {total_latency/successful:.2f}s")
        print(f"  Total Tokens: {total_tokens}")
    
    # Save results
    output_dir = "/mnt/fss/ARBM/benchmarks/results"
    os.makedirs(output_dir, exist_ok=True)
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "provider": "anthropic",
        "summary": {
            "total_tasks": len(TASKS),
            "successful": successful,
            "success_rate": successful/len(TASKS),
            "avg_latency": total_latency/successful if successful > 0 else 0,
            "total_tokens": total_tokens
        },
        "results": results
    }
    
    output_file = f"{output_dir}/arbm_api_benchmark_{datetime.now().strftime(\"%Y%m%d_%H%M%S\")}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n  Results saved to: {output_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
