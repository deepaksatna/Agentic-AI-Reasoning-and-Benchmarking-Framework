#!/usr/bin/env python3
"""
ARBM Track 07: Code Generation & Debugging Benchmarks
Evaluates ability to generate, understand, and fix code - SWE-bench style

Author: Deepak Soni
License: MIT
"""

import requests
import json
import time
import re
from datetime import datetime
from typing import Dict, List, Any
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1/chat/completions")
MODEL = os.environ.get("MODEL_PATH", "/mnt/fss/2026-NIM-vLLM_LLM/models/llama-3-8b-instruct")
OUTPUT_DIR = "/mnt/fss/ARBM/benchmarks/trending/track_07_code_gen/results"

# =============================================================================
# CODE GENERATION TASKS
# =============================================================================

CODE_GEN_TASKS = [
    {
        "id": "codegen_001",
        "name": "Fibonacci Function",
        "category": "algorithms",
        "prompt": "Write a Python function called 'fibonacci' that returns the nth Fibonacci number. Handle edge cases for n <= 0.",
        "test_cases": [
            {"input": "fibonacci(0)", "expected": "0"},
            {"input": "fibonacci(1)", "expected": "1"},
            {"input": "fibonacci(10)", "expected": "55"},
            {"input": "fibonacci(-1)", "expected": "0"}
        ],
        "keywords": ["def fibonacci", "return"]
    },
    {
        "id": "codegen_002",
        "name": "Binary Search",
        "category": "algorithms",
        "prompt": "Write a Python function called 'binary_search' that takes a sorted list and a target value, returns the index if found or -1 if not found.",
        "test_cases": [
            {"input": "binary_search([1,2,3,4,5], 3)", "expected": "2"},
            {"input": "binary_search([1,2,3,4,5], 6)", "expected": "-1"},
            {"input": "binary_search([], 1)", "expected": "-1"}
        ],
        "keywords": ["def binary_search", "while", "mid"]
    },
    {
        "id": "codegen_003",
        "name": "Palindrome Checker",
        "category": "strings",
        "prompt": "Write a Python function called 'is_palindrome' that checks if a string is a palindrome, ignoring case and non-alphanumeric characters.",
        "test_cases": [
            {"input": "is_palindrome('A man a plan a canal Panama')", "expected": "True"},
            {"input": "is_palindrome('race a car')", "expected": "False"},
            {"input": "is_palindrome('')", "expected": "True"}
        ],
        "keywords": ["def is_palindrome", "return"]
    },
    {
        "id": "codegen_004",
        "name": "Merge Two Sorted Lists",
        "category": "data_structures",
        "prompt": "Write a Python function called 'merge_sorted' that merges two sorted lists into one sorted list.",
        "test_cases": [
            {"input": "merge_sorted([1,3,5], [2,4,6])", "expected": "[1, 2, 3, 4, 5, 6]"},
            {"input": "merge_sorted([], [1,2])", "expected": "[1, 2]"},
            {"input": "merge_sorted([1], [])", "expected": "[1]"}
        ],
        "keywords": ["def merge_sorted", "while", "return"]
    },
    {
        "id": "codegen_005",
        "name": "LRU Cache",
        "category": "design",
        "prompt": "Write a Python class called 'LRUCache' with methods 'get(key)' and 'put(key, value)'. The cache should have a capacity limit.",
        "test_cases": [
            {"input": "cache = LRUCache(2); cache.put(1, 1); cache.get(1)", "expected": "1"},
            {"input": "cache = LRUCache(1); cache.put(1, 1); cache.put(2, 2); cache.get(1)", "expected": "-1"}
        ],
        "keywords": ["class LRUCache", "def get", "def put", "capacity"]
    }
]

# =============================================================================
# CODE DEBUGGING TASKS
# =============================================================================

DEBUG_TASKS = [
    {
        "id": "debug_001",
        "name": "Fix Off-by-One Error",
        "category": "bugs",
        "buggy_code": '''def sum_range(n):
    """Sum numbers from 1 to n"""
    total = 0
    for i in range(n):  # Bug: should be range(1, n+1)
        total += i
    return total''',
        "expected_fix": "range(1, n+1)",
        "description": "The function should sum numbers from 1 to n, but it's summing 0 to n-1"
    },
    {
        "id": "debug_002",
        "name": "Fix Infinite Loop",
        "category": "bugs",
        "buggy_code": '''def count_down(n):
    """Count down from n to 0"""
    while n > 0:
        print(n)
        # Bug: missing n -= 1
    return "Done"''',
        "expected_fix": "n -= 1",
        "description": "The function has an infinite loop because n is never decremented"
    },
    {
        "id": "debug_003",
        "name": "Fix Type Error",
        "category": "type_errors",
        "buggy_code": '''def concatenate_items(items):
    """Concatenate all items into a string"""
    result = ""
    for item in items:
        result += item  # Bug: item might not be a string
    return result''',
        "expected_fix": "str(item)",
        "description": "The function fails when items contain non-string types"
    },
    {
        "id": "debug_004",
        "name": "Fix Logic Error",
        "category": "logic",
        "buggy_code": '''def is_leap_year(year):
    """Check if year is a leap year"""
    if year % 4 == 0:  # Bug: incomplete leap year logic
        return True
    return False''',
        "expected_fix": "% 100",
        "description": "Leap year logic is incomplete - doesn't handle century years correctly"
    },
    {
        "id": "debug_005",
        "name": "Fix Index Error",
        "category": "bounds",
        "buggy_code": '''def get_neighbors(arr, idx):
    """Get neighbors of element at index"""
    left = arr[idx - 1]  # Bug: no bounds check
    right = arr[idx + 1]  # Bug: no bounds check
    return left, right''',
        "expected_fix": "if idx",
        "description": "The function doesn't handle boundary cases for first/last elements"
    }
]

# =============================================================================
# CODE EXPLANATION TASKS
# =============================================================================

EXPLAIN_TASKS = [
    {
        "id": "explain_001",
        "name": "Explain Recursion",
        "category": "explanation",
        "code": '''def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)''',
        "questions": ["What is the base case?", "What is the time complexity?"],
        "expected_keywords": ["base case", "n <= 1", "O(n)", "recursive"]
    },
    {
        "id": "explain_002",
        "name": "Explain Dynamic Programming",
        "category": "explanation",
        "code": '''def climb_stairs(n):
    if n <= 2:
        return n
    dp = [0] * (n + 1)
    dp[1], dp[2] = 1, 2
    for i in range(3, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]''',
        "questions": ["What pattern is this?", "Why use dp array?"],
        "expected_keywords": ["dynamic programming", "memoization", "fibonacci", "subproblem"]
    },
    {
        "id": "explain_003",
        "name": "Explain Two Pointers",
        "category": "explanation",
        "code": '''def two_sum_sorted(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:
        total = arr[left] + arr[right]
        if total == target:
            return [left, right]
        elif total < target:
            left += 1
        else:
            right -= 1
    return []''',
        "questions": ["What technique is used?", "Why is it efficient?"],
        "expected_keywords": ["two pointers", "O(n)", "sorted", "linear"]
    }
]

# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def call_llm(prompt: str, system: str = None, max_tokens: int = 800) -> Dict[str, Any]:
    """Call the LLM"""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        start_time = time.time()
        response = requests.post(
            VLLM_URL,
            json={
                "model": MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.1
            },
            timeout=120
        )
        latency = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "content": data["choices"][0]["message"]["content"],
                "latency": latency,
                "usage": data.get("usage", {})
            }
        return {"success": False, "error": response.text[:200], "latency": latency}
    except Exception as e:
        return {"success": False, "error": str(e)[:200], "latency": 0}


def run_code_generation_benchmark() -> Dict[str, Any]:
    """Run code generation benchmark"""
    print("\n  Running Code Generation benchmark...")

    results = []
    system = "You are an expert Python programmer. Write clean, efficient code with proper error handling."

    for task in CODE_GEN_TASKS:
        print(f"    [{task['id']}] {task['name']}...")

        response = call_llm(task["prompt"], system)

        if response["success"]:
            code = response["content"]
            # Check for required keywords
            keywords_found = sum(1 for kw in task["keywords"] if kw.lower() in code.lower())
            keyword_score = keywords_found / len(task["keywords"])

            # Extract code block if present
            code_match = re.search(r'```python\n(.*?)```', code, re.DOTALL)
            if code_match:
                extracted_code = code_match.group(1)
            else:
                extracted_code = code

            result = {
                "task_id": task["id"],
                "name": task["name"],
                "category": task["category"],
                "success": True,
                "keyword_score": keyword_score,
                "latency": response["latency"],
                "code_preview": extracted_code[:300]
            }
        else:
            result = {
                "task_id": task["id"],
                "name": task["name"],
                "category": task["category"],
                "success": False,
                "error": response.get("error"),
                "latency": response["latency"]
            }

        results.append(result)
        status = f"Keywords: {result.get('keyword_score', 0)*100:.0f}%" if result["success"] else "FAIL"
        print(f"      {status}")

    successful = [r for r in results if r["success"]]
    avg_keyword_score = sum(r["keyword_score"] for r in successful) / len(successful) if successful else 0

    summary = {
        "benchmark": "Code Generation",
        "total_tasks": len(CODE_GEN_TASKS),
        "successful": len(successful),
        "avg_keyword_score": avg_keyword_score,
        "avg_latency": sum(r["latency"] for r in results) / len(results)
    }

    return {"summary": summary, "results": results}


def run_debugging_benchmark() -> Dict[str, Any]:
    """Run code debugging benchmark"""
    print("\n  Running Code Debugging benchmark...")

    results = []
    system = "You are an expert debugger. Identify the bug in the code and provide the fix."

    for task in DEBUG_TASKS:
        print(f"    [{task['id']}] {task['name']}...")

        prompt = f"""Here is buggy code:

```python
{task['buggy_code']}
```

Problem description: {task['description']}

Identify the bug and provide the fix."""

        response = call_llm(prompt, system)

        if response["success"]:
            fix = response["content"]
            # Check if expected fix is mentioned
            fix_found = task["expected_fix"].lower() in fix.lower()

            result = {
                "task_id": task["id"],
                "name": task["name"],
                "category": task["category"],
                "success": True,
                "fix_found": fix_found,
                "latency": response["latency"],
                "fix_preview": fix[:300]
            }
        else:
            result = {
                "task_id": task["id"],
                "name": task["name"],
                "category": task["category"],
                "success": False,
                "fix_found": False,
                "latency": response["latency"]
            }

        results.append(result)
        status = "PASS" if result.get("fix_found") else "FAIL"
        print(f"      {status}")

    fix_rate = sum(1 for r in results if r.get("fix_found")) / len(results)

    summary = {
        "benchmark": "Code Debugging",
        "total_tasks": len(DEBUG_TASKS),
        "successful": sum(1 for r in results if r["success"]),
        "fix_rate": fix_rate,
        "avg_latency": sum(r["latency"] for r in results) / len(results)
    }

    return {"summary": summary, "results": results}


def run_explanation_benchmark() -> Dict[str, Any]:
    """Run code explanation benchmark"""
    print("\n  Running Code Explanation benchmark...")

    results = []
    system = "You are a programming teacher. Explain code clearly and accurately."

    for task in EXPLAIN_TASKS:
        print(f"    [{task['id']}] {task['name']}...")

        prompt = f"""Explain this code:

```python
{task['code']}
```

Questions to answer:
{chr(10).join(f"- {q}" for q in task['questions'])}"""

        response = call_llm(prompt, system)

        if response["success"]:
            explanation = response["content"].lower()
            keywords_found = sum(1 for kw in task["expected_keywords"] if kw.lower() in explanation)
            keyword_coverage = keywords_found / len(task["expected_keywords"])

            result = {
                "task_id": task["id"],
                "name": task["name"],
                "category": task["category"],
                "success": True,
                "keyword_coverage": keyword_coverage,
                "latency": response["latency"],
                "explanation_preview": response["content"][:300]
            }
        else:
            result = {
                "task_id": task["id"],
                "name": task["name"],
                "category": task["category"],
                "success": False,
                "keyword_coverage": 0,
                "latency": response["latency"]
            }

        results.append(result)
        print(f"      Coverage: {result['keyword_coverage']*100:.0f}%")

    avg_coverage = sum(r["keyword_coverage"] for r in results) / len(results)

    summary = {
        "benchmark": "Code Explanation",
        "total_tasks": len(EXPLAIN_TASKS),
        "successful": sum(1 for r in results if r["success"]),
        "avg_keyword_coverage": avg_coverage,
        "avg_latency": sum(r["latency"] for r in results) / len(results)
    }

    return {"summary": summary, "results": results}


def main():
    """Run Track 07 benchmarks"""
    print("=" * 70)
    print("  ARBM TRACK 07: CODE GENERATION & DEBUGGING BENCHMARKS")
    print("  Model: Llama-3-8B-Instruct via vLLM")
    print("=" * 70)

    gen_results = run_code_generation_benchmark()
    debug_results = run_debugging_benchmark()
    explain_results = run_explanation_benchmark()

    # Print summary
    print("\n" + "=" * 70)
    print("  TRACK 07 SUMMARY")
    print("=" * 70)

    print(f"\n  Code Generation:")
    s = gen_results["summary"]
    print(f"    Tasks: {s['successful']}/{s['total_tasks']}")
    print(f"    Avg Keyword Score: {s['avg_keyword_score']*100:.1f}%")

    print(f"\n  Code Debugging:")
    s = debug_results["summary"]
    print(f"    Tasks: {s['successful']}/{s['total_tasks']}")
    print(f"    Fix Rate: {s['fix_rate']*100:.1f}%")

    print(f"\n  Code Explanation:")
    s = explain_results["summary"]
    print(f"    Tasks: {s['successful']}/{s['total_tasks']}")
    print(f"    Keyword Coverage: {s['avg_keyword_coverage']*100:.1f}%")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = {
        "timestamp": datetime.now().isoformat(),
        "track": "07_code_generation",
        "model": "llama-3-8b-instruct",
        "benchmarks": {
            "code_generation": gen_results,
            "code_debugging": debug_results,
            "code_explanation": explain_results
        }
    }

    outfile = f"{OUTPUT_DIR}/track_07_{timestamp}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved: {outfile}")
    print("=" * 70)


if __name__ == "__main__":
    main()
