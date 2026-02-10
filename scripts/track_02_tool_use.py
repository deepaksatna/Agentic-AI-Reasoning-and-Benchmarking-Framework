#!/usr/bin/env python3
"""
ARBM Track 02: Tool-use Efficiency Benchmarks
Evaluates Web Search, Code Execution, and Memory Tool capabilities

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
OUTPUT_DIR = "/mnt/fss/ARBM/tracks/track_02_tool_use/results"

# =============================================================================
# TOOL DEFINITIONS (Simulated MCP-style tools)
# =============================================================================

AVAILABLE_TOOLS = {
    "web_search": {
        "description": "Search the web for information",
        "parameters": ["query"],
        "example": "web_search(query='Python best practices')"
    },
    "fetch_url": {
        "description": "Fetch content from a URL",
        "parameters": ["url"],
        "example": "fetch_url(url='https://example.com')"
    },
    "execute_code": {
        "description": "Execute Python code and return results",
        "parameters": ["code"],
        "example": "execute_code(code='print(2+2)')"
    },
    "read_file": {
        "description": "Read contents of a file",
        "parameters": ["path"],
        "example": "read_file(path='/data/file.txt')"
    },
    "write_file": {
        "description": "Write content to a file",
        "parameters": ["path", "content"],
        "example": "write_file(path='/data/out.txt', content='Hello')"
    },
    "memory_store": {
        "description": "Store information in memory for later retrieval",
        "parameters": ["key", "value"],
        "example": "memory_store(key='user_name', value='Alice')"
    },
    "memory_retrieve": {
        "description": "Retrieve information from memory",
        "parameters": ["key"],
        "example": "memory_retrieve(key='user_name')"
    },
    "calculator": {
        "description": "Perform mathematical calculations",
        "parameters": ["expression"],
        "example": "calculator(expression='sqrt(144) + 5^2')"
    }
}

# =============================================================================
# TOOL SELECTION BENCHMARK TASKS
# =============================================================================

TOOL_SELECTION_TASKS = [
    {
        "id": "tool_sel_001",
        "name": "Web Research",
        "instruction": "You need to find information about the latest Python version features.",
        "expected_tool": "web_search",
        "evaluation": "tool_match"
    },
    {
        "id": "tool_sel_002",
        "name": "Code Execution",
        "instruction": "Calculate the factorial of 10 by running Python code.",
        "expected_tool": "execute_code",
        "evaluation": "tool_match"
    },
    {
        "id": "tool_sel_003",
        "name": "File Reading",
        "instruction": "Read the configuration from /etc/app/config.yaml file.",
        "expected_tool": "read_file",
        "evaluation": "tool_match"
    },
    {
        "id": "tool_sel_004",
        "name": "Memory Storage",
        "instruction": "Remember that the user's preferred language is Python for future reference.",
        "expected_tool": "memory_store",
        "evaluation": "tool_match"
    },
    {
        "id": "tool_sel_005",
        "name": "Mathematical Calculation",
        "instruction": "What is the square root of 256 plus the cube of 3?",
        "expected_tool": "calculator",
        "evaluation": "tool_match"
    },
    {
        "id": "tool_sel_006",
        "name": "URL Fetching",
        "instruction": "Get the content from https://api.github.com/repos/python/cpython",
        "expected_tool": "fetch_url",
        "evaluation": "tool_match"
    },
    {
        "id": "tool_sel_007",
        "name": "File Writing",
        "instruction": "Save the analysis results to /output/results.json",
        "expected_tool": "write_file",
        "evaluation": "tool_match"
    },
    {
        "id": "tool_sel_008",
        "name": "Memory Retrieval",
        "instruction": "What was the user's preferred programming language that we stored earlier?",
        "expected_tool": "memory_retrieve",
        "evaluation": "tool_match"
    }
]

# =============================================================================
# TOOL PARAMETER BENCHMARK TASKS
# =============================================================================

TOOL_PARAMETER_TASKS = [
    {
        "id": "tool_param_001",
        "name": "Search Query Formation",
        "instruction": "Search for: How to implement binary search in Python with examples",
        "expected_tool": "web_search",
        "expected_params": {"query": "binary search Python"},
        "evaluation": "param_check"
    },
    {
        "id": "tool_param_002",
        "name": "Code Parameter",
        "instruction": "Run Python code to generate the first 10 Fibonacci numbers",
        "expected_tool": "execute_code",
        "expected_params": {"code": "fibonacci"},
        "evaluation": "param_contains"
    },
    {
        "id": "tool_param_003",
        "name": "File Path Handling",
        "instruction": "Read the data from the CSV file at /data/users/records.csv",
        "expected_tool": "read_file",
        "expected_params": {"path": "/data/users/records.csv"},
        "evaluation": "param_check"
    },
    {
        "id": "tool_param_004",
        "name": "Key-Value Storage",
        "instruction": "Store the API rate limit of 1000 requests per hour",
        "expected_tool": "memory_store",
        "expected_params": {"key": "rate_limit", "value": "1000"},
        "evaluation": "param_contains"
    }
]

# =============================================================================
# MULTI-TOOL WORKFLOW TASKS
# =============================================================================

MULTI_TOOL_TASKS = [
    {
        "id": "multi_tool_001",
        "name": "Research and Code",
        "instruction": """Complete this workflow:
1. Search for the formula to calculate compound interest
2. Write Python code to calculate compound interest for $1000 at 5% for 3 years
3. Execute the code and return the result""",
        "expected_tools": ["web_search", "execute_code"],
        "evaluation": "multi_tool"
    },
    {
        "id": "multi_tool_002",
        "name": "Read-Process-Write",
        "instruction": """Complete this workflow:
1. Read data from /input/numbers.txt
2. Calculate the sum and average using code
3. Write the results to /output/stats.txt""",
        "expected_tools": ["read_file", "execute_code", "write_file"],
        "evaluation": "multi_tool"
    },
    {
        "id": "multi_tool_003",
        "name": "Fetch-Store-Retrieve",
        "instruction": """Complete this workflow:
1. Fetch the JSON data from https://api.example.com/config
2. Store the important settings in memory
3. Later retrieve the stored settings when needed""",
        "expected_tools": ["fetch_url", "memory_store", "memory_retrieve"],
        "evaluation": "multi_tool"
    }
]

# =============================================================================
# ERROR HANDLING TASKS
# =============================================================================

ERROR_HANDLING_TASKS = [
    {
        "id": "error_001",
        "name": "File Not Found",
        "instruction": "Read /nonexistent/file.txt. If it doesn't exist, explain what alternative actions could be taken.",
        "expected_behavior": "suggest alternatives",
        "evaluation": "error_recovery"
    },
    {
        "id": "error_002",
        "name": "Invalid Code",
        "instruction": "Execute this Python code: print(undefined_variable). Handle any errors appropriately.",
        "expected_behavior": "explain error",
        "evaluation": "error_recovery"
    },
    {
        "id": "error_003",
        "name": "Network Failure",
        "instruction": "Fetch https://invalid-domain-12345.com. If it fails, suggest what to do next.",
        "expected_behavior": "retry or alternative",
        "evaluation": "error_recovery"
    }
]

# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def call_llm(prompt: str, system_prompt: str = None, max_tokens: int = 1024) -> Dict[str, Any]:
    """Call the LLM with optional system prompt."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
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
            timeout=180
        )
        latency = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "content": data["choices"][0]["message"]["content"],
                "latency": latency,
                "tokens_in": data.get("usage", {}).get("prompt_tokens", 0),
                "tokens_out": data.get("usage", {}).get("completion_tokens", 0)
            }
        else:
            return {"success": False, "error": response.text[:200], "latency": latency}
    except Exception as e:
        return {"success": False, "error": str(e)[:200], "latency": 0}


def get_tools_system_prompt() -> str:
    """Generate system prompt with available tools."""
    tools_desc = "You have access to the following tools:\n\n"
    for name, info in AVAILABLE_TOOLS.items():
        tools_desc += f"- {name}: {info['description']}\n"
        tools_desc += f"  Parameters: {info['parameters']}\n"
        tools_desc += f"  Example: {info['example']}\n\n"

    tools_desc += """When you need to use a tool, respond with:
TOOL: <tool_name>
PARAMS: <parameters as JSON>
REASON: <why you chose this tool>

If multiple tools are needed, list them in order."""

    return tools_desc


def extract_tool_from_response(response: str) -> Dict[str, Any]:
    """Extract tool selection from LLM response."""
    result = {"tool": None, "params": {}, "reason": ""}

    # Look for TOOL: pattern
    tool_match = re.search(r'TOOL:\s*(\w+)', response, re.IGNORECASE)
    if tool_match:
        result["tool"] = tool_match.group(1).lower()

    # Look for tool names mentioned
    if not result["tool"]:
        for tool_name in AVAILABLE_TOOLS.keys():
            if tool_name in response.lower():
                result["tool"] = tool_name
                break

    # Extract parameters
    params_match = re.search(r'PARAMS:\s*(\{[^}]+\})', response, re.IGNORECASE)
    if params_match:
        try:
            result["params"] = json.loads(params_match.group(1))
        except:
            pass

    # Extract reason
    reason_match = re.search(r'REASON:\s*(.+?)(?=TOOL:|PARAMS:|$)', response, re.IGNORECASE | re.DOTALL)
    if reason_match:
        result["reason"] = reason_match.group(1).strip()

    return result


def run_tool_selection_benchmark() -> Dict[str, Any]:
    """Run tool selection benchmark."""
    print("\n" + "=" * 70)
    print("  TRACK 02.1: TOOL SELECTION BENCHMARKS")
    print("=" * 70)

    system_prompt = get_tools_system_prompt()
    results = []

    for task in TOOL_SELECTION_TASKS:
        task_id = task["id"]
        print(f"\n[{task_id}] {task['name']}...")

        prompt = f"Task: {task['instruction']}\n\nWhich tool would you use and why?"
        response = call_llm(prompt, system_prompt)

        if response["success"]:
            extracted = extract_tool_from_response(response["content"])
            is_correct = extracted["tool"] == task["expected_tool"]

            print(f"  Expected: {task['expected_tool']} | Got: {extracted['tool']} | Correct: {is_correct}")

            result = {
                "task_id": task_id,
                "name": task["name"],
                "success": True,
                "correct": is_correct,
                "expected_tool": task["expected_tool"],
                "selected_tool": extracted["tool"],
                "latency": response["latency"],
                "tokens_in": response["tokens_in"],
                "tokens_out": response["tokens_out"]
            }
        else:
            print(f"  ERROR: {response['error']}")
            result = {"task_id": task_id, "success": False, "error": response["error"]}

        results.append(result)

    successful = [r for r in results if r.get("success", False)]
    correct = [r for r in successful if r.get("correct", False)]

    summary = {
        "benchmark": "Tool Selection",
        "total_tasks": len(TOOL_SELECTION_TASKS),
        "successful": len(successful),
        "correct": len(correct),
        "accuracy": len(correct) / len(successful) if successful else 0,
        "avg_latency": sum(r["latency"] for r in successful) / len(successful) if successful else 0
    }

    return {"summary": summary, "results": results}


def run_tool_parameter_benchmark() -> Dict[str, Any]:
    """Run tool parameter benchmark."""
    print("\n" + "=" * 70)
    print("  TRACK 02.2: TOOL PARAMETER BENCHMARKS")
    print("=" * 70)

    system_prompt = get_tools_system_prompt()
    results = []

    for task in TOOL_PARAMETER_TASKS:
        task_id = task["id"]
        print(f"\n[{task_id}] {task['name']}...")

        prompt = f"Task: {task['instruction']}\n\nSelect the appropriate tool and provide the exact parameters."
        response = call_llm(prompt, system_prompt)

        if response["success"]:
            extracted = extract_tool_from_response(response["content"])
            tool_correct = extracted["tool"] == task["expected_tool"]

            # Check parameters
            param_correct = False
            if task["evaluation"] == "param_contains":
                for key, value in task["expected_params"].items():
                    if value.lower() in response["content"].lower():
                        param_correct = True
                        break
            else:
                param_correct = all(
                    str(v).lower() in response["content"].lower()
                    for v in task["expected_params"].values()
                )

            print(f"  Tool: {tool_correct} | Params: {param_correct}")

            result = {
                "task_id": task_id,
                "name": task["name"],
                "success": True,
                "tool_correct": tool_correct,
                "params_correct": param_correct,
                "latency": response["latency"],
                "tokens_in": response["tokens_in"],
                "tokens_out": response["tokens_out"]
            }
        else:
            print(f"  ERROR: {response['error']}")
            result = {"task_id": task_id, "success": False, "error": response["error"]}

        results.append(result)

    successful = [r for r in results if r.get("success", False)]
    tool_correct = sum(1 for r in successful if r.get("tool_correct", False))
    param_correct = sum(1 for r in successful if r.get("params_correct", False))

    summary = {
        "benchmark": "Tool Parameters",
        "total_tasks": len(TOOL_PARAMETER_TASKS),
        "successful": len(successful),
        "tool_accuracy": tool_correct / len(successful) if successful else 0,
        "param_accuracy": param_correct / len(successful) if successful else 0,
        "avg_latency": sum(r["latency"] for r in successful) / len(successful) if successful else 0
    }

    return {"summary": summary, "results": results}


def run_multi_tool_benchmark() -> Dict[str, Any]:
    """Run multi-tool workflow benchmark."""
    print("\n" + "=" * 70)
    print("  TRACK 02.3: MULTI-TOOL WORKFLOW BENCHMARKS")
    print("=" * 70)

    system_prompt = get_tools_system_prompt()
    results = []

    for task in MULTI_TOOL_TASKS:
        task_id = task["id"]
        print(f"\n[{task_id}] {task['name']}...")

        prompt = f"Complete this multi-step task:\n\n{task['instruction']}\n\nList all tools you would use in order."
        response = call_llm(prompt, system_prompt, max_tokens=1500)

        if response["success"]:
            # Check which expected tools are mentioned
            tools_found = []
            for tool in task["expected_tools"]:
                if tool in response["content"].lower():
                    tools_found.append(tool)

            coverage = len(tools_found) / len(task["expected_tools"])
            print(f"  Tools Found: {tools_found} | Coverage: {coverage*100:.0f}%")

            result = {
                "task_id": task_id,
                "name": task["name"],
                "success": True,
                "expected_tools": task["expected_tools"],
                "tools_found": tools_found,
                "coverage": coverage,
                "latency": response["latency"],
                "tokens_in": response["tokens_in"],
                "tokens_out": response["tokens_out"]
            }
        else:
            print(f"  ERROR: {response['error']}")
            result = {"task_id": task_id, "success": False, "error": response["error"]}

        results.append(result)

    successful = [r for r in results if r.get("success", False)]
    avg_coverage = sum(r.get("coverage", 0) for r in successful) / len(successful) if successful else 0

    summary = {
        "benchmark": "Multi-Tool Workflows",
        "total_tasks": len(MULTI_TOOL_TASKS),
        "successful": len(successful),
        "avg_tool_coverage": avg_coverage,
        "avg_latency": sum(r["latency"] for r in successful) / len(successful) if successful else 0
    }

    return {"summary": summary, "results": results}


def run_error_handling_benchmark() -> Dict[str, Any]:
    """Run error handling benchmark."""
    print("\n" + "=" * 70)
    print("  TRACK 02.4: ERROR HANDLING BENCHMARKS")
    print("=" * 70)

    system_prompt = get_tools_system_prompt()
    results = []

    for task in ERROR_HANDLING_TASKS:
        task_id = task["id"]
        print(f"\n[{task_id}] {task['name']}...")

        prompt = f"Task: {task['instruction']}"
        response = call_llm(prompt, system_prompt)

        if response["success"]:
            # Check for error recovery indicators
            recovery_indicators = ["alternative", "instead", "retry", "error", "exception", "fail", "handle", "fallback"]
            recovery_found = any(ind in response["content"].lower() for ind in recovery_indicators)

            print(f"  Error Recovery Suggested: {recovery_found}")

            result = {
                "task_id": task_id,
                "name": task["name"],
                "success": True,
                "recovery_suggested": recovery_found,
                "latency": response["latency"],
                "tokens_in": response["tokens_in"],
                "tokens_out": response["tokens_out"]
            }
        else:
            print(f"  ERROR: {response['error']}")
            result = {"task_id": task_id, "success": False, "error": response["error"]}

        results.append(result)

    successful = [r for r in results if r.get("success", False)]
    recovery_rate = sum(1 for r in successful if r.get("recovery_suggested", False)) / len(successful) if successful else 0

    summary = {
        "benchmark": "Error Handling",
        "total_tasks": len(ERROR_HANDLING_TASKS),
        "successful": len(successful),
        "recovery_rate": recovery_rate,
        "avg_latency": sum(r["latency"] for r in successful) / len(successful) if successful else 0
    }

    return {"summary": summary, "results": results}


def main():
    """Run all Track 02 benchmarks."""
    print("=" * 70)
    print("  ARBM TRACK 02: TOOL-USE EFFICIENCY BENCHMARKS")
    print("  Model: Llama-3-8B-Instruct via vLLM")
    print("=" * 70)

    # Run all benchmarks
    selection_results = run_tool_selection_benchmark()
    parameter_results = run_tool_parameter_benchmark()
    multi_tool_results = run_multi_tool_benchmark()
    error_results = run_error_handling_benchmark()

    # Print summary
    print("\n" + "=" * 70)
    print("  TRACK 02 SUMMARY")
    print("=" * 70)

    for name, results in [
        ("Selection", selection_results),
        ("Parameters", parameter_results),
        ("Multi-Tool", multi_tool_results),
        ("Error Handling", error_results)
    ]:
        s = results["summary"]
        print(f"\n  {s['benchmark']}:")
        print(f"    Tasks: {s['total_tasks']} | Successful: {s['successful']}")
        if "accuracy" in s:
            print(f"    Accuracy: {s['accuracy']*100:.1f}%")
        if "avg_tool_coverage" in s:
            print(f"    Tool Coverage: {s['avg_tool_coverage']*100:.1f}%")
        if "recovery_rate" in s:
            print(f"    Recovery Rate: {s['recovery_rate']*100:.1f}%")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = {
        "timestamp": datetime.now().isoformat(),
        "track": "02_tool_use",
        "model": "llama-3-8b-instruct",
        "benchmarks": {
            "tool_selection": selection_results,
            "tool_parameters": parameter_results,
            "multi_tool_workflows": multi_tool_results,
            "error_handling": error_results
        }
    }

    outfile = f"{OUTPUT_DIR}/track_02_{timestamp}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved: {outfile}")
    print("=" * 70)


if __name__ == "__main__":
    main()
