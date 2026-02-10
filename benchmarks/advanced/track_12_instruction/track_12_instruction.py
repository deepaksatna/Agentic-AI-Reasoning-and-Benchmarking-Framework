#!/usr/bin/env python3
"""
ARBM Track 12: Instruction Following Benchmarks (IFEval style)
Evaluates ability to follow complex, multi-constraint instructions

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
OUTPUT_DIR = "/mnt/fss/ARBM/benchmarks/advanced/track_12_instruction/results"

# Format Constraint Tasks
FORMAT_TASKS = [
    {
        "id": "format_001",
        "name": "Word Count Constraint",
        "prompt": "Write exactly 50 words about artificial intelligence.",
        "check": lambda x: 45 <= len(x.split()) <= 55,
        "constraint": "50 words (±5)"
    },
    {
        "id": "format_002",
        "name": "Bullet Points",
        "prompt": "List exactly 5 benefits of exercise using bullet points.",
        "check": lambda x: x.count("•") == 5 or x.count("-") >= 5 or x.count("*") >= 5 or len(re.findall(r'^\d+\.', x, re.MULTILINE)) >= 5,
        "constraint": "5 bullet points"
    },
    {
        "id": "format_003",
        "name": "All Caps",
        "prompt": "Write a short warning message about fire safety. THE ENTIRE RESPONSE MUST BE IN UPPERCASE.",
        "check": lambda x: x.upper() == x or x.isupper(),
        "constraint": "All uppercase"
    },
    {
        "id": "format_004",
        "name": "No Commas",
        "prompt": "Describe your favorite food without using any commas in your response.",
        "check": lambda x: "," not in x,
        "constraint": "No commas"
    },
    {
        "id": "format_005",
        "name": "Numbered Steps",
        "prompt": "Explain how to make coffee in exactly 6 numbered steps (1. 2. 3. etc).",
        "check": lambda x: all(f"{i}." in x for i in range(1, 7)),
        "constraint": "6 numbered steps"
    },
    {
        "id": "format_006",
        "name": "First Letter Constraint",
        "prompt": "Write a sentence where every word starts with the letter 'S'.",
        "check": lambda x: all(w[0].lower() == 's' for w in x.split() if w.isalpha()),
        "constraint": "All words start with S"
    }
]

# Content Constraint Tasks
CONTENT_TASKS = [
    {
        "id": "content_001",
        "name": "Include Keywords",
        "prompt": "Write a paragraph about climate change. You MUST include these exact words: carbon, temperature, ocean, renewable.",
        "keywords": ["carbon", "temperature", "ocean", "renewable"],
        "check": lambda x, kw: all(k.lower() in x.lower() for k in kw),
        "constraint": "Must include 4 keywords"
    },
    {
        "id": "content_002",
        "name": "Exclude Words",
        "prompt": "Describe the color blue without using the words: sky, ocean, water, sea.",
        "forbidden": ["sky", "ocean", "water", "sea"],
        "check": lambda x, fw: not any(w.lower() in x.lower() for w in fw),
        "constraint": "Must not use forbidden words"
    },
    {
        "id": "content_003",
        "name": "Specific Structure",
        "prompt": "Write about dogs with exactly 3 paragraphs. Each paragraph must start with 'Dogs are'.",
        "check": lambda x: x.count("Dogs are") >= 3,
        "constraint": "3 paragraphs starting with 'Dogs are'"
    },
    {
        "id": "content_004",
        "name": "Question Ending",
        "prompt": "Explain photosynthesis. End your response with exactly 2 questions for the reader.",
        "check": lambda x: x.count("?") >= 2,
        "constraint": "End with 2 questions"
    }
]

# Multi-Constraint Tasks
MULTI_TASKS = [
    {
        "id": "multi_001",
        "name": "Three Constraints",
        "prompt": "Write about space exploration. Requirements: 1) Exactly 3 paragraphs 2) Include the word 'NASA' 3) End with an exclamation mark!",
        "checks": [
            ("3 paragraphs", lambda x: x.count("\n\n") >= 2 or len([p for p in x.split("\n") if p.strip()]) >= 3),
            ("Contains NASA", lambda x: "nasa" in x.lower()),
            ("Ends with !", lambda x: x.strip().endswith("!"))
        ]
    },
    {
        "id": "multi_002",
        "name": "Format + Content",
        "prompt": "List 4 programming languages as bullet points. Each must include the year it was created.",
        "checks": [
            ("4 items", lambda x: x.count("•") >= 4 or x.count("-") >= 4 or x.count("*") >= 4),
            ("Contains years", lambda x: len(re.findall(r'\b(19|20)\d{2}\b', x)) >= 3)
        ]
    },
    {
        "id": "multi_003",
        "name": "Complex Format",
        "prompt": "Create a haiku about technology (5-7-5 syllables). Title it 'Digital Dreams'. Put the title in quotes.",
        "checks": [
            ("Has title", lambda x: "digital dreams" in x.lower()),
            ("Title quoted", lambda x: '"' in x or "'" in x),
            ("3 lines poem", lambda x: len([l for l in x.split("\n") if l.strip() and "digital" not in l.lower()]) >= 3)
        ]
    },
    {
        "id": "multi_004",
        "name": "Precise Control",
        "prompt": "Write exactly 3 sentences about robots. Each sentence must be between 10-15 words. Do not use the word 'machine'.",
        "checks": [
            ("3 sentences", lambda x: x.count(".") >= 3),
            ("No 'machine'", lambda x: "machine" not in x.lower()),
            ("Reasonable length", lambda x: len(x.split()) >= 25)
        ]
    }
]

# Role Following Tasks
ROLE_TASKS = [
    {
        "id": "role_001",
        "name": "Pirate Speech",
        "prompt": "You are a pirate. Explain how computers work. Stay in character throughout - use pirate language and expressions.",
        "indicators": ["arr", "ye", "matey", "ahoy", "ship", "sea", "treasure", "captain"],
        "min_indicators": 2
    },
    {
        "id": "role_002",
        "name": "Formal Academic",
        "prompt": "You are a formal academic professor. Explain memes to someone. Use scholarly language, avoid slang.",
        "formal_indicators": ["furthermore", "therefore", "consequently", "phenomenon", "discourse", "scholarly", "academic", "research"],
        "min_indicators": 1
    },
    {
        "id": "role_003",
        "name": "Child-Friendly",
        "prompt": "You are explaining to a 5-year-old. Explain why the sky is blue. Use very simple words and fun comparisons.",
        "simple_check": lambda x: len([w for w in x.split() if len(w) > 10]) < 5,
        "constraint": "Simple language"
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


def run_format_benchmark() -> Dict[str, Any]:
    print("\n  Running Format Constraints benchmark...")
    results = []

    for task in FORMAT_TASKS:
        print(f"    [{task['id']}] {task['name']}...")
        response = call_llm(task["prompt"])

        if response["success"]:
            content = response["content"]
            constraint_met = task["check"](content)

            result = {
                "task_id": task["id"],
                "name": task["name"],
                "constraint": task["constraint"],
                "success": True,
                "constraint_met": constraint_met,
                "latency": response["latency"],
                "response_preview": content[:100]
            }
        else:
            result = {"task_id": task["id"], "success": False, "constraint_met": False}

        results.append(result)
        status = "PASS" if result.get("constraint_met") else "FAIL"
        print(f"      {status}")

    met = sum(1 for r in results if r.get("constraint_met"))
    return {
        "summary": {"benchmark": "Format Constraints", "total": len(FORMAT_TASKS), "compliance_rate": met / len(FORMAT_TASKS)},
        "results": results
    }


def run_content_benchmark() -> Dict[str, Any]:
    print("\n  Running Content Constraints benchmark...")
    results = []

    for task in CONTENT_TASKS:
        print(f"    [{task['id']}] {task['name']}...")
        response = call_llm(task["prompt"])

        if response["success"]:
            content = response["content"]
            if "keywords" in task:
                constraint_met = task["check"](content, task["keywords"])
            elif "forbidden" in task:
                constraint_met = task["check"](content, task["forbidden"])
            else:
                constraint_met = task["check"](content)

            result = {
                "task_id": task["id"],
                "name": task["name"],
                "constraint": task["constraint"],
                "success": True,
                "constraint_met": constraint_met,
                "latency": response["latency"]
            }
        else:
            result = {"task_id": task["id"], "success": False, "constraint_met": False}

        results.append(result)
        status = "PASS" if result.get("constraint_met") else "FAIL"
        print(f"      {status}")

    met = sum(1 for r in results if r.get("constraint_met"))
    return {
        "summary": {"benchmark": "Content Constraints", "total": len(CONTENT_TASKS), "compliance_rate": met / len(CONTENT_TASKS)},
        "results": results
    }


def run_multi_constraint_benchmark() -> Dict[str, Any]:
    print("\n  Running Multi-Constraint benchmark...")
    results = []

    for task in MULTI_TASKS:
        print(f"    [{task['id']}] {task['name']}...")
        response = call_llm(task["prompt"], max_tokens=600)

        if response["success"]:
            content = response["content"]
            check_results = []
            for name, check_fn in task["checks"]:
                passed = check_fn(content)
                check_results.append({"constraint": name, "passed": passed})

            constraints_met = sum(1 for c in check_results if c["passed"])
            total_constraints = len(task["checks"])

            result = {
                "task_id": task["id"],
                "name": task["name"],
                "success": True,
                "check_results": check_results,
                "constraints_met": constraints_met,
                "total_constraints": total_constraints,
                "score": constraints_met / total_constraints,
                "latency": response["latency"]
            }
        else:
            result = {"task_id": task["id"], "success": False, "score": 0}

        results.append(result)
        print(f"      {result.get('constraints_met', 0)}/{result.get('total_constraints', 0)} constraints")

    avg_score = sum(r.get("score", 0) for r in results) / len(results)
    return {
        "summary": {"benchmark": "Multi-Constraint", "total": len(MULTI_TASKS), "avg_compliance_score": avg_score},
        "results": results
    }


def run_role_benchmark() -> Dict[str, Any]:
    print("\n  Running Role Following benchmark...")
    results = []

    for task in ROLE_TASKS:
        print(f"    [{task['id']}] {task['name']}...")
        response = call_llm(task["prompt"], max_tokens=400)

        if response["success"]:
            content = response["content"].lower()

            if "indicators" in task:
                found = sum(1 for ind in task["indicators"] if ind in content)
                role_followed = found >= task["min_indicators"]
            elif "formal_indicators" in task:
                found = sum(1 for ind in task["formal_indicators"] if ind in content)
                role_followed = found >= task["min_indicators"]
            elif "simple_check" in task:
                role_followed = task["simple_check"](response["content"])
            else:
                role_followed = False

            result = {
                "task_id": task["id"],
                "name": task["name"],
                "success": True,
                "role_followed": role_followed,
                "latency": response["latency"],
                "response_preview": response["content"][:150]
            }
        else:
            result = {"task_id": task["id"], "success": False, "role_followed": False}

        results.append(result)
        status = "PASS" if result.get("role_followed") else "FAIL"
        print(f"      {status}")

    followed = sum(1 for r in results if r.get("role_followed"))
    return {
        "summary": {"benchmark": "Role Following", "total": len(ROLE_TASKS), "role_compliance_rate": followed / len(ROLE_TASKS)},
        "results": results
    }


def main():
    print("=" * 70)
    print("  ARBM TRACK 12: INSTRUCTION FOLLOWING BENCHMARKS")
    print("  Model: Llama-3-8B-Instruct via vLLM")
    print("=" * 70)

    format_results = run_format_benchmark()
    content_results = run_content_benchmark()
    multi_results = run_multi_constraint_benchmark()
    role_results = run_role_benchmark()

    print("\n" + "=" * 70)
    print("  TRACK 12 SUMMARY")
    print("=" * 70)
    print(f"\n  Format Constraints:  {format_results['summary']['compliance_rate']*100:.1f}%")
    print(f"  Content Constraints: {content_results['summary']['compliance_rate']*100:.1f}%")
    print(f"  Multi-Constraint:    {multi_results['summary']['avg_compliance_score']*100:.1f}%")
    print(f"  Role Following:      {role_results['summary']['role_compliance_rate']*100:.1f}%")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "track": "12_instruction_following",
        "model": "llama-3-8b-instruct",
        "benchmarks": {
            "format_constraints": format_results,
            "content_constraints": content_results,
            "multi_constraint": multi_results,
            "role_following": role_results
        }
    }

    outfile = f"{OUTPUT_DIR}/track_12_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved: {outfile}")
    print("=" * 70)


if __name__ == "__main__":
    main()
