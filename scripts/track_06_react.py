#!/usr/bin/env python3
"""
ARBM Track 06: ReAct (Reasoning + Acting) Benchmarks
Evaluates the ability to interleave reasoning with actions in an agent loop

Author: Deepak Soni
License: MIT
"""

import requests
import json
import time
import re
from datetime import datetime
from typing import Dict, List, Any, Tuple
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1/chat/completions")
MODEL = os.environ.get("MODEL_PATH", "/mnt/fss/2026-NIM-vLLM_LLM/models/llama-3-8b-instruct")
OUTPUT_DIR = "/mnt/fss/ARBM/benchmarks/trending/track_06_react/results"

# =============================================================================
# SIMULATED ENVIRONMENT FOR REACT
# =============================================================================

class SimulatedEnvironment:
    """Simulated environment for ReAct tasks"""

    def __init__(self):
        self.database = {
            "users": {
                "alice": {"age": 28, "city": "New York", "occupation": "Engineer"},
                "bob": {"age": 35, "city": "San Francisco", "occupation": "Designer"},
                "charlie": {"age": 42, "city": "Seattle", "occupation": "Manager"}
            },
            "products": {
                "laptop": {"price": 999, "stock": 50, "category": "electronics"},
                "phone": {"price": 699, "stock": 100, "category": "electronics"},
                "desk": {"price": 299, "stock": 25, "category": "furniture"}
            },
            "weather": {
                "New York": {"temp": 72, "condition": "sunny"},
                "San Francisco": {"temp": 65, "condition": "foggy"},
                "Seattle": {"temp": 58, "condition": "rainy"}
            }
        }
        self.action_log = []

    def execute_action(self, action: str, params: Dict) -> str:
        """Execute an action and return observation"""
        self.action_log.append({"action": action, "params": params})

        if action == "lookup_user":
            user = params.get("name", "").lower()
            if user in self.database["users"]:
                return f"Found user: {json.dumps(self.database['users'][user])}"
            return f"User '{user}' not found"

        elif action == "lookup_product":
            product = params.get("name", "").lower()
            if product in self.database["products"]:
                return f"Product info: {json.dumps(self.database['products'][product])}"
            return f"Product '{product}' not found"

        elif action == "check_weather":
            city = params.get("city", "")
            for c, data in self.database["weather"].items():
                if c.lower() == city.lower():
                    return f"Weather in {c}: {data['temp']}F, {data['condition']}"
            return f"Weather data not available for '{city}'"

        elif action == "calculate":
            expr = params.get("expression", "")
            try:
                result = eval(expr, {"__builtins__": {}}, {})
                return f"Calculation result: {result}"
            except:
                return "Calculation error"

        elif action == "search":
            query = params.get("query", "").lower()
            results = []
            for category, items in self.database.items():
                for name, data in items.items():
                    if query in name.lower() or query in str(data).lower():
                        results.append(f"{category}/{name}: {data}")
            return f"Search results: {results[:3]}" if results else "No results found"

        return f"Unknown action: {action}"


# =============================================================================
# REACT TASKS
# =============================================================================

REACT_TASKS = [
    {
        "id": "react_001",
        "name": "User Weather Lookup",
        "category": "multi_step",
        "question": "What is the weather like where Alice lives?",
        "expected_actions": ["lookup_user", "check_weather"],
        "expected_answer": "sunny",
        "max_steps": 5
    },
    {
        "id": "react_002",
        "name": "Price Comparison",
        "category": "multi_step",
        "question": "Is the laptop more expensive than the phone? By how much?",
        "expected_actions": ["lookup_product", "lookup_product", "calculate"],
        "expected_answer": "300",
        "max_steps": 6
    },
    {
        "id": "react_003",
        "name": "User Analysis",
        "category": "reasoning",
        "question": "Who is older, Alice or Bob? And what is the age difference?",
        "expected_actions": ["lookup_user", "lookup_user", "calculate"],
        "expected_answer": "7",
        "max_steps": 6
    },
    {
        "id": "react_004",
        "name": "Inventory Check",
        "category": "multi_step",
        "question": "What is the total value of all phone inventory (price * stock)?",
        "expected_actions": ["lookup_product", "calculate"],
        "expected_answer": "69900",
        "max_steps": 5
    },
    {
        "id": "react_005",
        "name": "Complex Query",
        "category": "complex",
        "question": "Find all users in cities where the weather is NOT sunny, and list their occupations.",
        "expected_actions": ["check_weather", "check_weather", "check_weather", "lookup_user"],
        "expected_answer": "Designer",
        "max_steps": 8
    },
    {
        "id": "react_006",
        "name": "Search and Reason",
        "category": "search",
        "question": "Search for 'electronics' and calculate the total price of all electronics products.",
        "expected_actions": ["search", "calculate"],
        "expected_answer": "1698",
        "max_steps": 5
    },
    {
        "id": "react_007",
        "name": "Chain Reasoning",
        "category": "chain",
        "question": "Bob lives in which city? What is the weather there? Is it warmer or colder than 70F?",
        "expected_actions": ["lookup_user", "check_weather"],
        "expected_answer": "colder",
        "max_steps": 6
    },
    {
        "id": "react_008",
        "name": "Data Aggregation",
        "category": "aggregation",
        "question": "What is the average age of all users in the database?",
        "expected_actions": ["lookup_user", "lookup_user", "lookup_user", "calculate"],
        "expected_answer": "35",
        "max_steps": 8
    }
]

# =============================================================================
# REACT AGENT
# =============================================================================

def call_llm(messages: List[Dict], max_tokens: int = 512) -> Dict[str, Any]:
    """Call the LLM"""
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


def parse_react_response(response: str) -> Tuple[str, str, Dict]:
    """Parse ReAct response into thought, action, and params"""
    thought = ""
    action = ""
    params = {}

    # Extract thought
    thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|$)', response, re.DOTALL | re.IGNORECASE)
    if thought_match:
        thought = thought_match.group(1).strip()

    # Extract action
    action_match = re.search(r'Action:\s*(\w+)', response, re.IGNORECASE)
    if action_match:
        action = action_match.group(1).lower()

    # Extract action input/params
    input_match = re.search(r'Action Input:\s*(.+?)(?=Thought:|Observation:|$)', response, re.DOTALL | re.IGNORECASE)
    if input_match:
        input_str = input_match.group(1).strip()
        # Try to parse as JSON
        try:
            params = json.loads(input_str)
        except:
            # Parse simple key=value or just value
            if "=" in input_str:
                for part in input_str.split(","):
                    if "=" in part:
                        k, v = part.split("=", 1)
                        params[k.strip()] = v.strip().strip('"\'')
            else:
                # Guess the parameter name based on action
                param_map = {
                    "lookup_user": "name",
                    "lookup_product": "name",
                    "check_weather": "city",
                    "calculate": "expression",
                    "search": "query"
                }
                params[param_map.get(action, "input")] = input_str.strip('"\'')

    # Check for final answer
    answer_match = re.search(r'Final Answer:\s*(.+?)$', response, re.DOTALL | re.IGNORECASE)
    if answer_match:
        action = "finish"
        params["answer"] = answer_match.group(1).strip()

    return thought, action, params


def run_react_agent(task: Dict, env: SimulatedEnvironment) -> Dict[str, Any]:
    """Run ReAct agent on a task"""

    system_prompt = """You are a ReAct agent that solves problems by interleaving Thought, Action, and Observation steps.

Available Actions:
- lookup_user: Look up user information. Action Input: {"name": "username"}
- lookup_product: Look up product information. Action Input: {"name": "productname"}
- check_weather: Check weather for a city. Action Input: {"city": "cityname"}
- calculate: Perform calculation. Action Input: {"expression": "math expression"}
- search: Search the database. Action Input: {"query": "search term"}

Format your response as:
Thought: [your reasoning about what to do next]
Action: [action name]
Action Input: [parameters as JSON]

When you have the final answer:
Thought: [final reasoning]
Final Answer: [your answer]

Always reason step by step and use actions to gather information before answering."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {task['question']}"}
    ]

    steps = []
    total_latency = 0
    actions_taken = []
    final_answer = None

    for step in range(task["max_steps"]):
        response = call_llm(messages, max_tokens=300)

        if not response["success"]:
            break

        total_latency += response["latency"]
        content = response["content"]

        thought, action, params = parse_react_response(content)

        step_data = {
            "step": step + 1,
            "thought": thought,
            "action": action,
            "params": params,
            "latency": response["latency"]
        }

        if action == "finish":
            final_answer = params.get("answer", "")
            step_data["final_answer"] = final_answer
            steps.append(step_data)
            break

        if action:
            actions_taken.append(action)
            observation = env.execute_action(action, params)
            step_data["observation"] = observation

            # Add observation to conversation
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        steps.append(step_data)

    # Calculate metrics
    expected_actions = set(task["expected_actions"])
    taken_actions = set(actions_taken)
    action_coverage = len(expected_actions & taken_actions) / len(expected_actions) if expected_actions else 0

    answer_correct = task["expected_answer"].lower() in (final_answer or "").lower()

    return {
        "task_id": task["id"],
        "category": task["category"],
        "name": task["name"],
        "question": task["question"],
        "steps": steps,
        "total_steps": len(steps),
        "actions_taken": actions_taken,
        "action_coverage": action_coverage,
        "final_answer": final_answer,
        "expected_answer": task["expected_answer"],
        "answer_correct": answer_correct,
        "total_latency": total_latency,
        "avg_step_latency": total_latency / len(steps) if steps else 0
    }


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_react_benchmark() -> Dict[str, Any]:
    """Run all ReAct benchmark tasks"""
    print("\n  Running ReAct (Reasoning + Acting) benchmark...")

    results = []
    env = SimulatedEnvironment()

    for task in REACT_TASKS:
        print(f"    [{task['id']}] {task['name']}...")
        env.action_log = []  # Reset action log

        result = run_react_agent(task, env)
        results.append(result)

        status = "PASS" if result["answer_correct"] else "FAIL"
        print(f"      Steps: {result['total_steps']}, Actions: {len(result['actions_taken'])}, {status}")

    # Calculate summary
    correct = sum(1 for r in results if r["answer_correct"])
    total = len(results)
    avg_steps = sum(r["total_steps"] for r in results) / total
    avg_actions = sum(len(r["actions_taken"]) for r in results) / total
    avg_coverage = sum(r["action_coverage"] for r in results) / total
    avg_latency = sum(r["total_latency"] for r in results) / total

    # Category breakdown
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"correct": 0, "total": 0}
        categories[cat]["total"] += 1
        if r["answer_correct"]:
            categories[cat]["correct"] += 1

    summary = {
        "benchmark": "ReAct (Reasoning + Acting)",
        "total_tasks": total,
        "correct": correct,
        "accuracy": correct / total,
        "avg_steps": avg_steps,
        "avg_actions": avg_actions,
        "avg_action_coverage": avg_coverage,
        "avg_latency": avg_latency,
        "category_accuracy": {k: v["correct"]/v["total"] for k, v in categories.items()}
    }

    return {"summary": summary, "results": results}


def main():
    """Run Track 06 benchmarks"""
    print("=" * 70)
    print("  ARBM TRACK 06: REACT (REASONING + ACTING) BENCHMARKS")
    print("  Model: Llama-3-8B-Instruct via vLLM")
    print("=" * 70)

    react_results = run_react_benchmark()

    # Print summary
    print("\n" + "=" * 70)
    print("  TRACK 06 SUMMARY")
    print("=" * 70)

    s = react_results["summary"]
    print(f"\n  ReAct Benchmark:")
    print(f"    Accuracy:           {s['accuracy']*100:.1f}% ({s['correct']}/{s['total_tasks']})")
    print(f"    Avg Steps:          {s['avg_steps']:.1f}")
    print(f"    Avg Actions:        {s['avg_actions']:.1f}")
    print(f"    Action Coverage:    {s['avg_action_coverage']*100:.1f}%")
    print(f"    Avg Latency:        {s['avg_latency']:.2f}s")

    print("\n  Category Accuracy:")
    for cat, acc in s["category_accuracy"].items():
        print(f"    {cat}: {acc*100:.1f}%")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = {
        "timestamp": datetime.now().isoformat(),
        "track": "06_react",
        "model": "llama-3-8b-instruct",
        "benchmarks": {"react": react_results}
    }

    outfile = f"{OUTPUT_DIR}/track_06_{timestamp}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved: {outfile}")
    print("=" * 70)


if __name__ == "__main__":
    main()
