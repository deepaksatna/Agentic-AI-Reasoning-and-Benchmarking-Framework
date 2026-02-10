#!/usr/bin/env python3
"""
ARBM Track 08: Self-Reflection & Self-Correction Benchmarks
Evaluates ability to identify errors in own reasoning and self-correct

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
OUTPUT_DIR = "/mnt/fss/ARBM/benchmarks/trending/track_08_self_reflect/results"

# =============================================================================
# SELF-REFLECTION TASKS
# =============================================================================

# Tasks where initial response is likely wrong and needs reflection
REFLECTION_TASKS = [
    {
        "id": "reflect_001",
        "name": "Math Trap Question",
        "category": "math",
        "question": "A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "common_wrong_answer": "10 cents",
        "correct_answer": "5 cents",
        "trap_explanation": "Most people intuitively say 10 cents, but $1.00 + $0.10 = $1.10, and $1.00 - $0.10 = $0.90, not $1.00 more"
    },
    {
        "id": "reflect_002",
        "name": "Logic Trap",
        "category": "logic",
        "question": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "common_wrong_answer": "100 minutes",
        "correct_answer": "5 minutes",
        "trap_explanation": "Each machine makes 1 widget in 5 minutes, so 100 machines make 100 widgets in 5 minutes"
    },
    {
        "id": "reflect_003",
        "name": "Survivorship Bias",
        "category": "reasoning",
        "question": "During WWII, engineers examined planes returning from combat. They found bullet holes concentrated on the wings and tail. Where should they add armor?",
        "common_wrong_answer": "wings and tail",
        "correct_answer": "cockpit and engines",
        "trap_explanation": "Survivorship bias - planes hit in wings/tail survived. Planes hit elsewhere didn't return."
    },
    {
        "id": "reflect_004",
        "name": "Probability Trap",
        "category": "probability",
        "question": "You flip a fair coin 5 times and get heads each time. What's the probability the next flip is heads?",
        "common_wrong_answer": "less than 50%",
        "correct_answer": "50%",
        "trap_explanation": "Gambler's fallacy - each flip is independent, previous results don't affect future"
    },
    {
        "id": "reflect_005",
        "name": "Cognitive Overload",
        "category": "calculation",
        "question": "You're running a race and pass the person in 2nd place. What place are you now?",
        "common_wrong_answer": "1st place",
        "correct_answer": "2nd place",
        "trap_explanation": "If you pass 2nd place, you take their position (2nd), not the one ahead (1st)"
    }
]

# Tasks for self-correction after feedback
CORRECTION_TASKS = [
    {
        "id": "correct_001",
        "name": "Factual Error Correction",
        "category": "facts",
        "initial_question": "What is the capital of Australia?",
        "common_error": "Sydney",
        "correction_prompt": "That's incorrect. Sydney is the largest city but not the capital. Can you reconsider?",
        "correct_answer": "Canberra"
    },
    {
        "id": "correct_002",
        "name": "Code Bug Correction",
        "category": "code",
        "initial_question": "Write Python to sum 1 to n: sum = 0; for i in range(n): sum += i",
        "common_error": "The code sums 0 to n-1",
        "correction_prompt": "Your code sums 0 to n-1, not 1 to n. Can you fix it?",
        "correct_answer": "range(1, n+1)"
    },
    {
        "id": "correct_003",
        "name": "Logic Correction",
        "category": "logic",
        "initial_question": "All roses are flowers. Some flowers fade quickly. Therefore, some roses fade quickly. Is this valid?",
        "common_error": "Yes, it's valid",
        "correction_prompt": "Consider: just because some flowers fade quickly, does it mean those flowers are roses? Reconsider.",
        "correct_answer": "invalid"
    }
]

# Tasks for iterative refinement
REFINEMENT_TASKS = [
    {
        "id": "refine_001",
        "name": "Essay Improvement",
        "category": "writing",
        "initial_prompt": "Write a one-sentence summary of climate change.",
        "refinement_prompts": [
            "Make it more specific about causes.",
            "Add the impact on ecosystems.",
            "Make it under 30 words while keeping key points."
        ],
        "quality_keywords": ["greenhouse", "temperature", "human", "ecosystem", "impact"]
    },
    {
        "id": "refine_002",
        "name": "Code Optimization",
        "category": "code",
        "initial_prompt": "Write a function to find duplicates in a list.",
        "refinement_prompts": [
            "Optimize for O(n) time complexity.",
            "Handle edge cases like empty lists.",
            "Add type hints."
        ],
        "quality_keywords": ["set", "O(n)", "if not", "List", "->"]
    }
]

# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def call_llm(messages: List[Dict], max_tokens: int = 500) -> Dict[str, Any]:
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
                "latency": latency
            }
        return {"success": False, "error": response.text[:200], "latency": latency}
    except Exception as e:
        return {"success": False, "error": str(e)[:200], "latency": 0}


def run_reflection_benchmark() -> Dict[str, Any]:
    """Run self-reflection benchmark with trap questions"""
    print("\n  Running Self-Reflection benchmark (trap questions)...")

    results = []

    for task in REFLECTION_TASKS:
        print(f"    [{task['id']}] {task['name']}...")

        # Step 1: Initial response
        messages = [{"role": "user", "content": task["question"]}]
        initial_response = call_llm(messages)

        if not initial_response["success"]:
            results.append({"task_id": task["id"], "success": False})
            continue

        initial_answer = initial_response["content"]
        fell_for_trap = task["common_wrong_answer"].lower() in initial_answer.lower()
        initial_correct = task["correct_answer"].lower() in initial_answer.lower()

        # Step 2: Ask for reflection
        messages.append({"role": "assistant", "content": initial_answer})
        messages.append({
            "role": "user",
            "content": "Wait, are you sure? Please double-check your answer step by step. Think carefully about any assumptions you might be making."
        })

        reflection_response = call_llm(messages)

        if not reflection_response["success"]:
            results.append({"task_id": task["id"], "success": False})
            continue

        reflected_answer = reflection_response["content"]
        reflected_correct = task["correct_answer"].lower() in reflected_answer.lower()

        # Determine if self-correction occurred
        self_corrected = fell_for_trap and reflected_correct

        result = {
            "task_id": task["id"],
            "name": task["name"],
            "category": task["category"],
            "success": True,
            "initial_correct": initial_correct,
            "fell_for_trap": fell_for_trap,
            "reflected_correct": reflected_correct,
            "self_corrected": self_corrected,
            "total_latency": initial_response["latency"] + reflection_response["latency"],
            "initial_preview": initial_answer[:150],
            "reflected_preview": reflected_answer[:150]
        }
        results.append(result)

        status = "CORRECTED" if self_corrected else ("CORRECT" if initial_correct else "WRONG")
        print(f"      Initial: {'Trap' if fell_for_trap else 'OK'}, After Reflection: {status}")

    # Calculate metrics
    successful = [r for r in results if r["success"]]
    initial_correct_rate = sum(1 for r in successful if r["initial_correct"]) / len(successful) if successful else 0
    trap_rate = sum(1 for r in successful if r["fell_for_trap"]) / len(successful) if successful else 0
    reflection_correct_rate = sum(1 for r in successful if r["reflected_correct"]) / len(successful) if successful else 0
    self_correction_rate = sum(1 for r in successful if r["self_corrected"]) / len(successful) if successful else 0

    summary = {
        "benchmark": "Self-Reflection",
        "total_tasks": len(REFLECTION_TASKS),
        "successful": len(successful),
        "initial_correct_rate": initial_correct_rate,
        "trap_rate": trap_rate,
        "reflection_correct_rate": reflection_correct_rate,
        "self_correction_rate": self_correction_rate,
        "improvement": reflection_correct_rate - initial_correct_rate
    }

    return {"summary": summary, "results": results}


def run_correction_benchmark() -> Dict[str, Any]:
    """Run self-correction benchmark with explicit feedback"""
    print("\n  Running Self-Correction benchmark (with feedback)...")

    results = []

    for task in CORRECTION_TASKS:
        print(f"    [{task['id']}] {task['name']}...")

        # Step 1: Initial response
        messages = [{"role": "user", "content": task["initial_question"]}]
        initial_response = call_llm(messages)

        if not initial_response["success"]:
            results.append({"task_id": task["id"], "success": False})
            continue

        initial_answer = initial_response["content"]
        initial_has_error = task["common_error"].lower() in initial_answer.lower()

        # Step 2: Provide correction feedback
        messages.append({"role": "assistant", "content": initial_answer})
        messages.append({"role": "user", "content": task["correction_prompt"]})

        corrected_response = call_llm(messages)

        if not corrected_response["success"]:
            results.append({"task_id": task["id"], "success": False})
            continue

        corrected_answer = corrected_response["content"]
        accepted_correction = task["correct_answer"].lower() in corrected_answer.lower()

        result = {
            "task_id": task["id"],
            "name": task["name"],
            "category": task["category"],
            "success": True,
            "initial_had_error": initial_has_error,
            "accepted_correction": accepted_correction,
            "total_latency": initial_response["latency"] + corrected_response["latency"],
            "initial_preview": initial_answer[:150],
            "corrected_preview": corrected_answer[:150]
        }
        results.append(result)

        status = "ACCEPTED" if accepted_correction else "REJECTED"
        print(f"      Correction {status}")

    successful = [r for r in results if r["success"]]
    correction_acceptance_rate = sum(1 for r in successful if r["accepted_correction"]) / len(successful) if successful else 0

    summary = {
        "benchmark": "Self-Correction",
        "total_tasks": len(CORRECTION_TASKS),
        "successful": len(successful),
        "correction_acceptance_rate": correction_acceptance_rate
    }

    return {"summary": summary, "results": results}


def run_refinement_benchmark() -> Dict[str, Any]:
    """Run iterative refinement benchmark"""
    print("\n  Running Iterative Refinement benchmark...")

    results = []

    for task in REFINEMENT_TASKS:
        print(f"    [{task['id']}] {task['name']}...")

        messages = [{"role": "user", "content": task["initial_prompt"]}]
        iterations = []
        total_latency = 0

        # Initial response
        response = call_llm(messages)
        if not response["success"]:
            results.append({"task_id": task["id"], "success": False})
            continue

        current_output = response["content"]
        total_latency += response["latency"]

        initial_keywords = sum(1 for kw in task["quality_keywords"] if kw.lower() in current_output.lower())
        iterations.append({
            "iteration": 0,
            "output_preview": current_output[:200],
            "keywords_found": initial_keywords
        })

        messages.append({"role": "assistant", "content": current_output})

        # Refinement iterations
        for i, refinement_prompt in enumerate(task["refinement_prompts"]):
            messages.append({"role": "user", "content": refinement_prompt})

            response = call_llm(messages)
            if not response["success"]:
                break

            current_output = response["content"]
            total_latency += response["latency"]

            keywords_found = sum(1 for kw in task["quality_keywords"] if kw.lower() in current_output.lower())
            iterations.append({
                "iteration": i + 1,
                "refinement": refinement_prompt,
                "output_preview": current_output[:200],
                "keywords_found": keywords_found
            })

            messages.append({"role": "assistant", "content": current_output})

        # Calculate improvement
        if len(iterations) >= 2:
            initial_score = iterations[0]["keywords_found"]
            final_score = iterations[-1]["keywords_found"]
            improvement = (final_score - initial_score) / len(task["quality_keywords"])
        else:
            improvement = 0

        result = {
            "task_id": task["id"],
            "name": task["name"],
            "category": task["category"],
            "success": True,
            "iterations": iterations,
            "num_iterations": len(iterations),
            "quality_improvement": improvement,
            "final_keyword_score": iterations[-1]["keywords_found"] / len(task["quality_keywords"]) if iterations else 0,
            "total_latency": total_latency
        }
        results.append(result)

        print(f"      Iterations: {len(iterations)}, Improvement: {improvement*100:.1f}%")

    successful = [r for r in results if r["success"]]
    avg_improvement = sum(r["quality_improvement"] for r in successful) / len(successful) if successful else 0
    avg_final_score = sum(r["final_keyword_score"] for r in successful) / len(successful) if successful else 0

    summary = {
        "benchmark": "Iterative Refinement",
        "total_tasks": len(REFINEMENT_TASKS),
        "successful": len(successful),
        "avg_quality_improvement": avg_improvement,
        "avg_final_keyword_score": avg_final_score
    }

    return {"summary": summary, "results": results}


def main():
    """Run Track 08 benchmarks"""
    print("=" * 70)
    print("  ARBM TRACK 08: SELF-REFLECTION & SELF-CORRECTION BENCHMARKS")
    print("  Model: Llama-3-8B-Instruct via vLLM")
    print("=" * 70)

    reflection_results = run_reflection_benchmark()
    correction_results = run_correction_benchmark()
    refinement_results = run_refinement_benchmark()

    # Print summary
    print("\n" + "=" * 70)
    print("  TRACK 08 SUMMARY")
    print("=" * 70)

    print("\n  Self-Reflection (Trap Questions):")
    s = reflection_results["summary"]
    print(f"    Initial Correct Rate:    {s['initial_correct_rate']*100:.1f}%")
    print(f"    Trap Rate:               {s['trap_rate']*100:.1f}%")
    print(f"    After Reflection:        {s['reflection_correct_rate']*100:.1f}%")
    print(f"    Self-Correction Rate:    {s['self_correction_rate']*100:.1f}%")
    print(f"    Improvement:             {s['improvement']*100:+.1f}%")

    print("\n  Self-Correction (With Feedback):")
    s = correction_results["summary"]
    print(f"    Correction Acceptance:   {s['correction_acceptance_rate']*100:.1f}%")

    print("\n  Iterative Refinement:")
    s = refinement_results["summary"]
    print(f"    Avg Quality Improvement: {s['avg_quality_improvement']*100:.1f}%")
    print(f"    Final Keyword Score:     {s['avg_final_keyword_score']*100:.1f}%")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = {
        "timestamp": datetime.now().isoformat(),
        "track": "08_self_reflection",
        "model": "llama-3-8b-instruct",
        "benchmarks": {
            "self_reflection": reflection_results,
            "self_correction": correction_results,
            "iterative_refinement": refinement_results
        }
    }

    outfile = f"{OUTPUT_DIR}/track_08_{timestamp}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved: {outfile}")
    print("=" * 70)


if __name__ == "__main__":
    main()
