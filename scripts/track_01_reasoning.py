#!/usr/bin/env python3
"""
ARBM Track 01: Reasoning Quality Benchmarks
Evaluates Chain-of-Thought (CoT), Tree of Thoughts (ToT), and Graph of Thoughts (GoT)

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
OUTPUT_DIR = "/mnt/fss/ARBM/tracks/track_01_reasoning/results"

# =============================================================================
# CHAIN-OF-THOUGHT (CoT) BENCHMARK TASKS
# =============================================================================

COT_TASKS = [
    # Mathematical Reasoning (GSM8K-style)
    {
        "id": "cot_math_001",
        "category": "mathematical",
        "name": "Multi-step Arithmetic",
        "prompt": "A bakery sells cupcakes for $3 each. If they make 48 cupcakes and sell 75% of them, how much revenue do they generate? Think step by step.",
        "expected_answer": "108",
        "evaluation": "exact_match"
    },
    {
        "id": "cot_math_002",
        "category": "mathematical",
        "name": "Rate Problem",
        "prompt": "Train A travels at 60 mph and Train B travels at 80 mph. If they start 280 miles apart and travel toward each other, how many hours until they meet? Show your reasoning.",
        "expected_answer": "2",
        "evaluation": "exact_match"
    },
    {
        "id": "cot_math_003",
        "category": "mathematical",
        "name": "Percentage Calculation",
        "prompt": "A store offers a 20% discount on a $150 item. Then, there's an additional 10% off the discounted price. What is the final price? Think step by step.",
        "expected_answer": "108",
        "evaluation": "exact_match"
    },
    {
        "id": "cot_math_004",
        "category": "mathematical",
        "name": "Compound Interest",
        "prompt": "If you invest $1000 at 10% annual interest compounded yearly, how much will you have after 2 years? Round to the nearest dollar. Show your work.",
        "expected_answer": "1210",
        "evaluation": "exact_match"
    },
    # Logical Reasoning
    {
        "id": "cot_logic_001",
        "category": "logical",
        "name": "Syllogism",
        "prompt": "If all programmers use computers, and some computer users are gamers, can we conclude that some programmers are gamers? Answer yes or no and explain your reasoning.",
        "expected_answer": "no",
        "evaluation": "contains"
    },
    {
        "id": "cot_logic_002",
        "category": "logical",
        "name": "Conditional Logic",
        "prompt": "If it rains, the ground gets wet. The ground is wet. Can we conclude it rained? Answer yes or no and explain why.",
        "expected_answer": "no",
        "evaluation": "contains"
    },
    {
        "id": "cot_logic_003",
        "category": "logical",
        "name": "Contrapositive",
        "prompt": "If a number is divisible by 6, it is divisible by 3. A number is not divisible by 3. What can we conclude? Think step by step.",
        "expected_answer": "not divisible by 6",
        "evaluation": "contains"
    },
    # Multi-hop Reasoning (HotpotQA-style)
    {
        "id": "cot_multihop_001",
        "category": "multi_hop",
        "name": "Height Comparison",
        "prompt": "Alice is taller than Bob. Bob is taller than Charlie. David is shorter than Charlie. Who is the tallest? Explain your reasoning.",
        "expected_answer": "Alice",
        "evaluation": "contains"
    },
    {
        "id": "cot_multihop_002",
        "category": "multi_hop",
        "name": "Age Ordering",
        "prompt": "Emma is older than Frank. Frank is younger than Grace. Grace is older than Emma. Order them from oldest to youngest.",
        "expected_answer": "Grace",
        "evaluation": "contains"
    },
    {
        "id": "cot_multihop_003",
        "category": "multi_hop",
        "name": "Location Reasoning",
        "prompt": "The library is north of the park. The school is south of the park. The hospital is north of the library. What is the order from south to north?",
        "expected_answer": "school",
        "evaluation": "contains"
    },
    # Causal Reasoning
    {
        "id": "cot_causal_001",
        "category": "causal",
        "name": "Cause and Effect",
        "prompt": "A plant's leaves turned yellow. It was placed in a dark room for a week. What is the most likely cause of the yellowing? Explain the biological process.",
        "expected_answer": "light",
        "evaluation": "contains"
    },
    # Temporal Reasoning
    {
        "id": "cot_temporal_001",
        "category": "temporal",
        "name": "Event Ordering",
        "prompt": "John ate breakfast at 8am. He left for work 30 minutes after eating. His commute takes 45 minutes. What time did he arrive at work?",
        "expected_answer": "9:15",
        "evaluation": "contains"
    }
]

# =============================================================================
# TREE OF THOUGHTS (ToT) BENCHMARK TASKS
# =============================================================================

TOT_TASKS = [
    {
        "id": "tot_puzzle_001",
        "category": "puzzle",
        "name": "Box Label Puzzle",
        "prompt": """You have 3 boxes labeled 'Apples', 'Oranges', and 'Mixed'. All labels are WRONG.
You can only pick ONE fruit from ONE box to correctly label ALL boxes.

Think through multiple possible approaches:
- Approach 1: Pick from 'Apples' box
- Approach 2: Pick from 'Oranges' box
- Approach 3: Pick from 'Mixed' box

Evaluate each approach and determine which one allows you to correctly label all boxes.""",
        "expected_answer": "Mixed",
        "evaluation": "contains"
    },
    {
        "id": "tot_puzzle_002",
        "category": "puzzle",
        "name": "River Crossing",
        "prompt": """A farmer needs to cross a river with a wolf, a goat, and a cabbage. The boat can only carry the farmer and one item.
If left alone: wolf eats goat, goat eats cabbage.

Explore different sequences:
- What happens if we take the goat first?
- What happens if we take the wolf first?
- What happens if we take the cabbage first?

Find a valid sequence of crossings.""",
        "expected_answer": "goat",
        "evaluation": "contains"
    },
    {
        "id": "tot_puzzle_003",
        "category": "puzzle",
        "name": "Game of 24",
        "prompt": """Using the numbers 8, 3, 8, 3 and the operations +, -, *, /, make the number 24.
Each number must be used exactly once.

Explore different combinations:
- Try (8 + 8) * (3 - 3/something)
- Try 8 * 3 - something
- Try 8 / (3 - 8/3)

Show your exploration and find a solution.""",
        "expected_answer": "24",
        "evaluation": "contains"
    },
    {
        "id": "tot_puzzle_004",
        "category": "puzzle",
        "name": "Knights and Knaves",
        "prompt": """On an island, knights always tell the truth and knaves always lie.
Person A says: "We are both knaves."
Person B says nothing.

Explore the possibilities:
- If A is a knight, what does that imply?
- If A is a knave, what does that imply?

Determine what A and B are.""",
        "expected_answer": "knave",
        "evaluation": "contains"
    },
    {
        "id": "tot_strategy_001",
        "category": "strategy",
        "name": "Optimal Path",
        "prompt": """You need to visit 4 cities: A, B, C, D starting from A.
Distances: A-B=10, A-C=15, A-D=20, B-C=35, B-D=25, C-D=30

Explore different routes:
- A -> B -> C -> D
- A -> B -> D -> C
- A -> C -> B -> D
- A -> C -> D -> B

Find the shortest route that visits all cities starting from A.""",
        "expected_answer": "65",
        "evaluation": "contains"
    }
]

# =============================================================================
# GRAPH OF THOUGHTS (GoT) BENCHMARK TASKS
# =============================================================================

GOT_TASKS = [
    {
        "id": "got_synthesis_001",
        "category": "synthesis",
        "name": "Multi-source Analysis",
        "prompt": """Analyze the following information from multiple sources about a new technology:

Source 1 (Tech Blog): "The new AI chip is 10x faster than previous generation"
Source 2 (Research Paper): "Power consumption increased by 30% compared to predecessor"
Source 3 (Industry Report): "Manufacturing costs are 2x higher but yield rates improved by 50%"
Source 4 (User Review): "Real-world performance shows 7x speedup in typical workloads"

Connect these pieces of information:
- How do the claims relate to each other?
- Are there contradictions?
- What's the overall assessment?

Provide a synthesized analysis.""",
        "expected_answer": "trade",
        "evaluation": "contains"
    },
    {
        "id": "got_dependency_001",
        "category": "dependency",
        "name": "Task Dependencies",
        "prompt": """A software project has these tasks and dependencies:
- Task A: No dependencies (takes 2 days)
- Task B: Depends on A (takes 3 days)
- Task C: Depends on A (takes 4 days)
- Task D: Depends on B and C (takes 2 days)
- Task E: Depends on C (takes 1 day)
- Task F: Depends on D and E (takes 3 days)

Create a dependency graph and determine:
1. The critical path
2. Minimum project duration
3. Which tasks can be parallelized""",
        "expected_answer": "12",
        "evaluation": "contains"
    },
    {
        "id": "got_reasoning_001",
        "category": "reasoning_web",
        "name": "Connected Reasoning",
        "prompt": """Consider these interconnected facts:
1. All mammals are warm-blooded
2. Whales are mammals
3. Fish are cold-blooded
4. Dolphins look like fish but are mammals
5. Some sea creatures that look similar have different body temperatures

From these facts, create a reasoning graph that answers:
- Why do dolphins and sharks, despite looking similar, have different characteristics?
- What common misconception might arise from visual similarity?""",
        "expected_answer": "warm-blooded",
        "evaluation": "contains"
    },
    {
        "id": "got_causal_001",
        "category": "causal_graph",
        "name": "Causal Network",
        "prompt": """Analyze this causal network for a business scenario:
- Economic downturn -> Reduced consumer spending
- Reduced consumer spending -> Lower company revenue
- Lower company revenue -> Staff layoffs
- Staff layoffs -> Reduced productivity
- Reduced productivity -> Lower quality products
- Lower quality products -> Further reduced revenue (feedback loop)
- Economic recovery -> Increased consumer confidence
- Increased consumer confidence -> Higher spending

Identify:
1. The main feedback loops
2. Potential intervention points
3. How breaking one link affects the system""",
        "expected_answer": "feedback",
        "evaluation": "contains"
    }
]

# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def call_llm(prompt: str, max_tokens: int = 1024) -> Dict[str, Any]:
    """Call the LLM and return response with metrics."""
    try:
        start_time = time.time()
        response = requests.post(
            VLLM_URL,
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
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


def evaluate_response(response: str, expected: str, eval_type: str) -> bool:
    """Evaluate if the response matches expected answer."""
    response_lower = response.lower()
    expected_lower = expected.lower()

    if eval_type == "exact_match":
        # Extract numbers from response
        numbers = re.findall(r'\b\d+\.?\d*\b', response)
        return expected_lower in [n.lower() for n in numbers]
    elif eval_type == "contains":
        return expected_lower in response_lower
    return False


def count_reasoning_steps(response: str) -> int:
    """Count the number of reasoning steps in the response."""
    step_patterns = [
        r'step \d+',
        r'\d+\.',
        r'first,|second,|third,|then,|finally,|next,',
        r'therefore|thus|hence|so,',
    ]
    count = 0
    for pattern in step_patterns:
        count += len(re.findall(pattern, response.lower()))
    return max(1, count)


def run_cot_benchmark() -> Dict[str, Any]:
    """Run Chain-of-Thought benchmark."""
    print("\n" + "=" * 70)
    print("  TRACK 01.1: CHAIN-OF-THOUGHT (CoT) BENCHMARKS")
    print("=" * 70)

    results = []
    categories = {}

    for task in COT_TASKS:
        task_id = task["id"]
        print(f"\n[{task_id}] {task['name']} ({task['category']})...")

        response = call_llm(task["prompt"])

        if response["success"]:
            is_correct = evaluate_response(response["content"], task["expected_answer"], task["evaluation"])
            reasoning_steps = count_reasoning_steps(response["content"])

            print(f"  Latency: {response['latency']:.2f}s | Correct: {is_correct} | Steps: {reasoning_steps}")

            result = {
                "task_id": task_id,
                "category": task["category"],
                "name": task["name"],
                "success": True,
                "correct": is_correct,
                "latency": response["latency"],
                "tokens_in": response["tokens_in"],
                "tokens_out": response["tokens_out"],
                "reasoning_steps": reasoning_steps,
                "response_preview": response["content"][:200]
            }

            # Track by category
            if task["category"] not in categories:
                categories[task["category"]] = {"correct": 0, "total": 0}
            categories[task["category"]]["total"] += 1
            if is_correct:
                categories[task["category"]]["correct"] += 1
        else:
            print(f"  ERROR: {response['error']}")
            result = {"task_id": task_id, "success": False, "error": response["error"]}

        results.append(result)

    # Calculate summary
    successful = [r for r in results if r.get("success", False)]
    correct = [r for r in successful if r.get("correct", False)]

    summary = {
        "benchmark": "Chain-of-Thought",
        "total_tasks": len(COT_TASKS),
        "successful": len(successful),
        "correct": len(correct),
        "accuracy": len(correct) / len(successful) if successful else 0,
        "avg_latency": sum(r["latency"] for r in successful) / len(successful) if successful else 0,
        "avg_reasoning_steps": sum(r.get("reasoning_steps", 0) for r in successful) / len(successful) if successful else 0,
        "category_breakdown": {k: v["correct"]/v["total"] for k, v in categories.items()}
    }

    return {"summary": summary, "results": results}


def run_tot_benchmark() -> Dict[str, Any]:
    """Run Tree of Thoughts benchmark."""
    print("\n" + "=" * 70)
    print("  TRACK 01.2: TREE OF THOUGHTS (ToT) BENCHMARKS")
    print("=" * 70)

    results = []

    for task in TOT_TASKS:
        task_id = task["id"]
        print(f"\n[{task_id}] {task['name']} ({task['category']})...")

        response = call_llm(task["prompt"], max_tokens=1500)

        if response["success"]:
            is_correct = evaluate_response(response["content"], task["expected_answer"], task["evaluation"])

            # Count exploration branches
            branch_patterns = [r'approach \d+', r'option \d+', r'try \d+', r'possibility \d+', r'if we']
            branches = sum(len(re.findall(p, response["content"].lower())) for p in branch_patterns)

            print(f"  Latency: {response['latency']:.2f}s | Correct: {is_correct} | Branches: {branches}")

            result = {
                "task_id": task_id,
                "category": task["category"],
                "name": task["name"],
                "success": True,
                "correct": is_correct,
                "latency": response["latency"],
                "tokens_in": response["tokens_in"],
                "tokens_out": response["tokens_out"],
                "exploration_branches": branches,
                "response_preview": response["content"][:200]
            }
        else:
            print(f"  ERROR: {response['error']}")
            result = {"task_id": task_id, "success": False, "error": response["error"]}

        results.append(result)

    successful = [r for r in results if r.get("success", False)]
    correct = [r for r in successful if r.get("correct", False)]

    summary = {
        "benchmark": "Tree-of-Thoughts",
        "total_tasks": len(TOT_TASKS),
        "successful": len(successful),
        "correct": len(correct),
        "accuracy": len(correct) / len(successful) if successful else 0,
        "avg_latency": sum(r["latency"] for r in successful) / len(successful) if successful else 0,
        "avg_branches": sum(r.get("exploration_branches", 0) for r in successful) / len(successful) if successful else 0
    }

    return {"summary": summary, "results": results}


def run_got_benchmark() -> Dict[str, Any]:
    """Run Graph of Thoughts benchmark."""
    print("\n" + "=" * 70)
    print("  TRACK 01.3: GRAPH OF THOUGHTS (GoT) BENCHMARKS")
    print("=" * 70)

    results = []

    for task in GOT_TASKS:
        task_id = task["id"]
        print(f"\n[{task_id}] {task['name']} ({task['category']})...")

        response = call_llm(task["prompt"], max_tokens=2000)

        if response["success"]:
            is_correct = evaluate_response(response["content"], task["expected_answer"], task["evaluation"])

            # Analyze graph complexity
            connection_words = ['connects', 'relates', 'leads to', 'causes', 'depends on', 'linked', 'between']
            connections = sum(response["content"].lower().count(w) for w in connection_words)

            print(f"  Latency: {response['latency']:.2f}s | Correct: {is_correct} | Connections: {connections}")

            result = {
                "task_id": task_id,
                "category": task["category"],
                "name": task["name"],
                "success": True,
                "correct": is_correct,
                "latency": response["latency"],
                "tokens_in": response["tokens_in"],
                "tokens_out": response["tokens_out"],
                "graph_connections": connections,
                "response_preview": response["content"][:200]
            }
        else:
            print(f"  ERROR: {response['error']}")
            result = {"task_id": task_id, "success": False, "error": response["error"]}

        results.append(result)

    successful = [r for r in results if r.get("success", False)]
    correct = [r for r in successful if r.get("correct", False)]

    summary = {
        "benchmark": "Graph-of-Thoughts",
        "total_tasks": len(GOT_TASKS),
        "successful": len(successful),
        "correct": len(correct),
        "accuracy": len(correct) / len(successful) if successful else 0,
        "avg_latency": sum(r["latency"] for r in successful) / len(successful) if successful else 0,
        "avg_connections": sum(r.get("graph_connections", 0) for r in successful) / len(successful) if successful else 0
    }

    return {"summary": summary, "results": results}


def main():
    """Run all Track 01 benchmarks."""
    print("=" * 70)
    print("  ARBM TRACK 01: REASONING QUALITY BENCHMARKS")
    print("  Model: Llama-3-8B-Instruct via vLLM")
    print("=" * 70)

    # Run all benchmarks
    cot_results = run_cot_benchmark()
    tot_results = run_tot_benchmark()
    got_results = run_got_benchmark()

    # Print summary
    print("\n" + "=" * 70)
    print("  TRACK 01 SUMMARY")
    print("=" * 70)

    for name, results in [("CoT", cot_results), ("ToT", tot_results), ("GoT", got_results)]:
        s = results["summary"]
        print(f"\n  {s['benchmark']}:")
        print(f"    Tasks: {s['total_tasks']} | Correct: {s['correct']}/{s['successful']}")
        print(f"    Accuracy: {s['accuracy']*100:.1f}% | Avg Latency: {s['avg_latency']:.2f}s")

    # Calculate overall
    total_correct = cot_results["summary"]["correct"] + tot_results["summary"]["correct"] + got_results["summary"]["correct"]
    total_tasks = cot_results["summary"]["successful"] + tot_results["summary"]["successful"] + got_results["summary"]["successful"]

    print(f"\n  OVERALL TRACK 01:")
    print(f"    Total Correct: {total_correct}/{total_tasks}")
    print(f"    Overall Accuracy: {total_correct/total_tasks*100:.1f}%" if total_tasks > 0 else "N/A")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = {
        "timestamp": datetime.now().isoformat(),
        "track": "01_reasoning_quality",
        "model": "llama-3-8b-instruct",
        "benchmarks": {
            "chain_of_thought": cot_results,
            "tree_of_thoughts": tot_results,
            "graph_of_thoughts": got_results
        },
        "overall": {
            "total_tasks": total_tasks,
            "total_correct": total_correct,
            "overall_accuracy": total_correct/total_tasks if total_tasks > 0 else 0
        }
    }

    outfile = f"{OUTPUT_DIR}/track_01_{timestamp}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved: {outfile}")
    print("=" * 70)


if __name__ == "__main__":
    main()
