#!/usr/bin/env python3
"""
ARBM Track 13: Math & STEM Reasoning Benchmarks (GSM8K style)
Evaluates mathematical problem solving and STEM reasoning

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

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1/chat/completions")
MODEL = os.environ.get("MODEL_PATH", "/mnt/fss/2026-NIM-vLLM_LLM/models/llama-3-8b-instruct")
OUTPUT_DIR = "/mnt/fss/ARBM/benchmarks/advanced/track_13_math_stem/results"

# GSM8K Style Word Problems
MATH_WORD_PROBLEMS = [
    {
        "id": "gsm_001",
        "name": "Shopping Problem",
        "problem": "Sarah has $50. She buys 3 books at $8 each and 2 pens at $3 each. How much money does she have left?",
        "answer": 20,
        "solution_steps": ["3 * 8 = 24", "2 * 3 = 6", "24 + 6 = 30", "50 - 30 = 20"]
    },
    {
        "id": "gsm_002",
        "name": "Distance Problem",
        "problem": "A car travels at 60 mph for 2.5 hours, then at 40 mph for 1.5 hours. What is the total distance traveled?",
        "answer": 210,
        "solution_steps": ["60 * 2.5 = 150", "40 * 1.5 = 60", "150 + 60 = 210"]
    },
    {
        "id": "gsm_003",
        "name": "Age Problem",
        "problem": "Tom is twice as old as his sister. In 5 years, the sum of their ages will be 40. How old is Tom now?",
        "answer": 20,
        "solution_steps": ["Let sister = x, Tom = 2x", "(x+5) + (2x+5) = 40", "3x + 10 = 40", "x = 10, Tom = 20"]
    },
    {
        "id": "gsm_004",
        "name": "Work Rate Problem",
        "problem": "John can paint a room in 6 hours. Mary can paint it in 4 hours. How long will it take them working together?",
        "answer": 2.4,
        "solution_steps": ["John rate = 1/6", "Mary rate = 1/4", "Combined = 1/6 + 1/4 = 5/12", "Time = 12/5 = 2.4"]
    },
    {
        "id": "gsm_005",
        "name": "Percentage Problem",
        "problem": "A store increases prices by 20%, then offers a 15% discount. If the original price was $100, what is the final price?",
        "answer": 102,
        "solution_steps": ["After increase: 100 * 1.20 = 120", "After discount: 120 * 0.85 = 102"]
    },
    {
        "id": "gsm_006",
        "name": "Mixture Problem",
        "problem": "How many liters of 30% acid solution must be mixed with 20 liters of 50% acid solution to get a 40% acid solution?",
        "answer": 20,
        "solution_steps": ["0.3x + 0.5(20) = 0.4(x+20)", "0.3x + 10 = 0.4x + 8", "2 = 0.1x", "x = 20"]
    }
]

# Algebra Problems
ALGEBRA_PROBLEMS = [
    {
        "id": "alg_001",
        "name": "Linear Equation",
        "problem": "Solve for x: 3x + 7 = 22",
        "answer": 5,
        "type": "equation"
    },
    {
        "id": "alg_002",
        "name": "Quadratic",
        "problem": "Find the positive value of x: x² - 5x + 6 = 0",
        "answer": 3,  # or 2, both valid
        "alt_answer": 2,
        "type": "equation"
    },
    {
        "id": "alg_003",
        "name": "System of Equations",
        "problem": "Solve: 2x + y = 10 and x - y = 2. What is the value of x?",
        "answer": 4,
        "type": "system"
    },
    {
        "id": "alg_004",
        "name": "Inequality",
        "problem": "What is the smallest integer x that satisfies: 3x - 4 > 11?",
        "answer": 6,
        "type": "inequality"
    }
]

# Science Reasoning Problems
SCIENCE_PROBLEMS = [
    {
        "id": "sci_001",
        "name": "Physics - Speed",
        "problem": "A ball is dropped from 80 meters. Using g=10 m/s², how long does it take to hit the ground? (Use h = 0.5*g*t²)",
        "answer": 4,
        "formula": "t = sqrt(2h/g)"
    },
    {
        "id": "sci_002",
        "name": "Chemistry - Moles",
        "problem": "How many moles of H2O are in 90 grams of water? (Molar mass of H2O = 18 g/mol)",
        "answer": 5,
        "formula": "moles = mass / molar_mass"
    },
    {
        "id": "sci_003",
        "name": "Physics - Energy",
        "problem": "What is the kinetic energy of a 2 kg object moving at 5 m/s? (KE = 0.5*m*v²)",
        "answer": 25,
        "formula": "KE = 0.5 * m * v^2"
    },
    {
        "id": "sci_004",
        "name": "Biology - Genetics",
        "problem": "In a cross between two heterozygous parents (Aa x Aa), what percentage of offspring will be homozygous?",
        "answer": 50,
        "explanation": "AA (25%) + aa (25%) = 50% homozygous"
    }
]

# Logic & Reasoning Problems
LOGIC_PROBLEMS = [
    {
        "id": "logic_001",
        "name": "Number Sequence",
        "problem": "What is the next number in the sequence: 2, 6, 12, 20, 30, ?",
        "answer": 42,
        "pattern": "n(n+1): differences are 4, 6, 8, 10, 12"
    },
    {
        "id": "logic_002",
        "name": "Probability",
        "problem": "A bag has 4 red and 6 blue balls. What is the probability (as percentage) of drawing 2 red balls in a row without replacement?",
        "answer": 13,  # 4/10 * 3/9 = 12/90 ≈ 13.3%
        "calculation": "(4/10) * (3/9) = 12/90 ≈ 13.3%"
    },
    {
        "id": "logic_003",
        "name": "Combinatorics",
        "problem": "How many ways can you arrange 3 books on a shelf from a collection of 5 books?",
        "answer": 60,
        "formula": "P(5,3) = 5!/(5-3)! = 60"
    }
]


def call_llm(prompt: str, system: str = None, max_tokens: int = 500) -> Dict[str, Any]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        start_time = time.time()
        response = requests.post(
            VLLM_URL,
            json={"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.1},
            timeout=120
        )
        latency = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            return {"success": True, "content": data["choices"][0]["message"]["content"], "latency": latency}
        return {"success": False, "error": response.text[:200], "latency": latency}
    except Exception as e:
        return {"success": False, "error": str(e)[:200], "latency": 0}


def extract_number(text: str) -> float:
    """Extract the final numerical answer from text"""
    # Look for patterns like "answer is X", "= X", "X dollars", etc.
    patterns = [
        r'(?:answer|result|total|value|solution)(?:\s+is)?[:\s]+\$?([-+]?\d*\.?\d+)',
        r'=\s*\$?([-+]?\d*\.?\d+)\s*(?:$|[.\n])',
        r'\$?([-+]?\d*\.?\d+)\s*(?:dollars?|hours?|meters?|liters?|%|percent)',
        r'(?:^|\n)\s*\$?([-+]?\d*\.?\d+)\s*$',
        r'\*\*([-+]?\d*\.?\d+)\*\*',
        r'boxed{([-+]?\d*\.?\d+)}'
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            try:
                return float(matches[-1])
            except:
                continue

    # Fallback: find all numbers and return the last one
    numbers = re.findall(r'[-+]?\d*\.?\d+', text)
    if numbers:
        try:
            return float(numbers[-1])
        except:
            pass

    return None


def check_answer(extracted: float, expected: float, alt: float = None, tolerance: float = 0.1) -> bool:
    """Check if extracted answer matches expected (with tolerance)"""
    if extracted is None:
        return False

    if abs(extracted - expected) <= tolerance:
        return True
    if alt is not None and abs(extracted - alt) <= tolerance:
        return True

    # Check percentage vs decimal confusion
    if abs(extracted - expected * 100) <= tolerance or abs(extracted / 100 - expected) <= tolerance:
        return True

    return False


def run_word_problems_benchmark() -> Dict[str, Any]:
    print("\n  Running GSM8K-style Word Problems...")
    results = []
    system = "You are a math tutor. Solve the problem step by step and clearly state the final numerical answer."

    for task in MATH_WORD_PROBLEMS:
        print(f"    [{task['id']}] {task['name']}...")
        response = call_llm(task["problem"], system)

        if response["success"]:
            extracted = extract_number(response["content"])
            correct = check_answer(extracted, task["answer"])

            result = {
                "task_id": task["id"],
                "name": task["name"],
                "success": True,
                "extracted_answer": extracted,
                "expected_answer": task["answer"],
                "correct": correct,
                "latency": response["latency"],
                "response_preview": response["content"][:200]
            }
        else:
            result = {"task_id": task["id"], "success": False, "correct": False}

        results.append(result)
        status = "CORRECT" if result.get("correct") else f"WRONG (got {result.get('extracted_answer')})"
        print(f"      {status}")

    correct = sum(1 for r in results if r.get("correct"))
    return {
        "summary": {"benchmark": "Word Problems (GSM8K)", "total": len(MATH_WORD_PROBLEMS), "accuracy": correct / len(MATH_WORD_PROBLEMS)},
        "results": results
    }


def run_algebra_benchmark() -> Dict[str, Any]:
    print("\n  Running Algebra Problems...")
    results = []
    system = "Solve the math problem. Show your work and provide the numerical answer."

    for task in ALGEBRA_PROBLEMS:
        print(f"    [{task['id']}] {task['name']}...")
        response = call_llm(task["problem"], system)

        if response["success"]:
            extracted = extract_number(response["content"])
            correct = check_answer(extracted, task["answer"], task.get("alt_answer"))

            result = {
                "task_id": task["id"],
                "name": task["name"],
                "success": True,
                "extracted_answer": extracted,
                "expected_answer": task["answer"],
                "correct": correct,
                "latency": response["latency"]
            }
        else:
            result = {"task_id": task["id"], "success": False, "correct": False}

        results.append(result)
        status = "CORRECT" if result.get("correct") else "WRONG"
        print(f"      {status}")

    correct = sum(1 for r in results if r.get("correct"))
    return {
        "summary": {"benchmark": "Algebra", "total": len(ALGEBRA_PROBLEMS), "accuracy": correct / len(ALGEBRA_PROBLEMS)},
        "results": results
    }


def run_science_benchmark() -> Dict[str, Any]:
    print("\n  Running Science Problems...")
    results = []
    system = "You are a science teacher. Solve the problem using the appropriate formula and show your calculation."

    for task in SCIENCE_PROBLEMS:
        print(f"    [{task['id']}] {task['name']}...")
        response = call_llm(task["problem"], system)

        if response["success"]:
            extracted = extract_number(response["content"])
            correct = check_answer(extracted, task["answer"], tolerance=1)

            result = {
                "task_id": task["id"],
                "name": task["name"],
                "success": True,
                "extracted_answer": extracted,
                "expected_answer": task["answer"],
                "correct": correct,
                "latency": response["latency"]
            }
        else:
            result = {"task_id": task["id"], "success": False, "correct": False}

        results.append(result)
        status = "CORRECT" if result.get("correct") else "WRONG"
        print(f"      {status}")

    correct = sum(1 for r in results if r.get("correct"))
    return {
        "summary": {"benchmark": "Science", "total": len(SCIENCE_PROBLEMS), "accuracy": correct / len(SCIENCE_PROBLEMS)},
        "results": results
    }


def run_logic_benchmark() -> Dict[str, Any]:
    print("\n  Running Logic & Reasoning Problems...")
    results = []
    system = "Solve the problem logically. Explain your reasoning and provide the numerical answer."

    for task in LOGIC_PROBLEMS:
        print(f"    [{task['id']}] {task['name']}...")
        response = call_llm(task["problem"], system)

        if response["success"]:
            extracted = extract_number(response["content"])
            correct = check_answer(extracted, task["answer"], tolerance=1)

            result = {
                "task_id": task["id"],
                "name": task["name"],
                "success": True,
                "extracted_answer": extracted,
                "expected_answer": task["answer"],
                "correct": correct,
                "latency": response["latency"]
            }
        else:
            result = {"task_id": task["id"], "success": False, "correct": False}

        results.append(result)
        status = "CORRECT" if result.get("correct") else "WRONG"
        print(f"      {status}")

    correct = sum(1 for r in results if r.get("correct"))
    return {
        "summary": {"benchmark": "Logic & Reasoning", "total": len(LOGIC_PROBLEMS), "accuracy": correct / len(LOGIC_PROBLEMS)},
        "results": results
    }


def main():
    print("=" * 70)
    print("  ARBM TRACK 13: MATH & STEM REASONING BENCHMARKS")
    print("  Model: Llama-3-8B-Instruct via vLLM")
    print("=" * 70)

    word_results = run_word_problems_benchmark()
    algebra_results = run_algebra_benchmark()
    science_results = run_science_benchmark()
    logic_results = run_logic_benchmark()

    print("\n" + "=" * 70)
    print("  TRACK 13 SUMMARY")
    print("=" * 70)
    print(f"\n  Word Problems (GSM8K): {word_results['summary']['accuracy']*100:.1f}%")
    print(f"  Algebra:               {algebra_results['summary']['accuracy']*100:.1f}%")
    print(f"  Science:               {science_results['summary']['accuracy']*100:.1f}%")
    print(f"  Logic & Reasoning:     {logic_results['summary']['accuracy']*100:.1f}%")

    total_correct = (word_results['summary']['accuracy'] + algebra_results['summary']['accuracy'] +
                     science_results['summary']['accuracy'] + logic_results['summary']['accuracy']) / 4
    print(f"\n  Overall STEM Score:    {total_correct*100:.1f}%")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "track": "13_math_stem",
        "model": "llama-3-8b-instruct",
        "benchmarks": {
            "word_problems": word_results,
            "algebra": algebra_results,
            "science": science_results,
            "logic": logic_results
        },
        "overall_score": total_correct
    }

    outfile = f"{OUTPUT_DIR}/track_13_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved: {outfile}")
    print("=" * 70)


if __name__ == "__main__":
    main()
