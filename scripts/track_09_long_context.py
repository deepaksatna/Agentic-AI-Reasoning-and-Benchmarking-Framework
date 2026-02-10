#!/usr/bin/env python3
"""
ARBM Track 09: Long-Context & Memory Management Benchmarks
Evaluates ability to handle long contexts, maintain memory, and retrieve information

Author: Deepak Soni
License: MIT
"""

import requests
import json
import time
import re
import random
from datetime import datetime
from typing import Dict, List, Any
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1/chat/completions")
MODEL = os.environ.get("MODEL_PATH", "/mnt/fss/2026-NIM-vLLM_LLM/models/llama-3-8b-instruct")
OUTPUT_DIR = "/mnt/fss/ARBM/benchmarks/trending/track_09_long_context/results"

# =============================================================================
# NEEDLE IN HAYSTACK TASKS
# =============================================================================

def generate_haystack(needle: str, needle_position: float, total_sentences: int = 50) -> str:
    """Generate a haystack with a needle at specified position"""
    filler_sentences = [
        "The weather today is particularly pleasant with clear skies.",
        "Technology continues to advance at an unprecedented rate.",
        "Many people enjoy spending their weekends outdoors.",
        "The stock market showed mixed results this quarter.",
        "Scientists are making progress in renewable energy research.",
        "Education plays a crucial role in personal development.",
        "Healthcare systems around the world face various challenges.",
        "Art and culture contribute significantly to society.",
        "Transportation infrastructure requires constant maintenance.",
        "Environmental conservation efforts are increasingly important.",
        "Sports bring communities together and promote fitness.",
        "Music has the power to evoke strong emotions.",
        "Literature provides windows into different perspectives.",
        "Architecture reflects the values of different eras.",
        "Cuisine varies greatly across different cultures.",
        "Fashion trends change with each passing season.",
        "Social media has transformed how we communicate.",
        "Space exploration continues to reveal new discoveries.",
        "Marine biology studies reveal fascinating ocean life.",
        "Psychology helps us understand human behavior better."
    ]

    sentences = []
    needle_index = int(total_sentences * needle_position)

    for i in range(total_sentences):
        if i == needle_index:
            sentences.append(needle)
        else:
            sentences.append(random.choice(filler_sentences))

    return " ".join(sentences)


NEEDLE_TASKS = [
    {
        "id": "needle_001",
        "name": "Beginning Needle",
        "needle": "The secret password is: ALPHA-BETA-GAMMA.",
        "question": "What is the secret password mentioned in the text?",
        "expected": "ALPHA-BETA-GAMMA",
        "position": 0.1
    },
    {
        "id": "needle_002",
        "name": "Middle Needle",
        "needle": "The CEO's birthday is on July 15th, 1985.",
        "question": "When is the CEO's birthday?",
        "expected": "July 15",
        "position": 0.5
    },
    {
        "id": "needle_003",
        "name": "End Needle",
        "needle": "The project deadline is December 31st, 2026.",
        "question": "What is the project deadline?",
        "expected": "December 31",
        "position": 0.9
    },
    {
        "id": "needle_004",
        "name": "Hidden Number",
        "needle": "For reference, use code 7429 when contacting support.",
        "question": "What code should be used when contacting support?",
        "expected": "7429",
        "position": 0.3
    },
    {
        "id": "needle_005",
        "name": "Name Retrieval",
        "needle": "The lead researcher, Dr. Sarah Mitchell, published groundbreaking results.",
        "question": "Who is the lead researcher mentioned?",
        "expected": "Sarah Mitchell",
        "position": 0.7
    }
]

# =============================================================================
# MULTI-HOP CONTEXT TASKS
# =============================================================================

MULTI_HOP_CONTEXTS = [
    {
        "id": "multihop_001",
        "name": "Person-Location-Event",
        "context": """
John works at TechCorp in Building A. Building A is located on Oak Street.
The annual company meeting is held at the main conference room.
The main conference room is on the third floor of Building A.
TechCorp's CEO, Maria, will present the quarterly results.
The meeting is scheduled for 3 PM on Friday.
""",
        "question": "Where will John need to go for the meeting and when?",
        "expected_elements": ["Building A", "third floor", "3 PM", "Friday"]
    },
    {
        "id": "multihop_002",
        "name": "Chain of Ownership",
        "context": """
Alice owns a blue car. Bob borrowed Alice's car last week.
Carol needed a car, so Bob lent it to her.
Carol drove to Denver for a conference.
The conference was about renewable energy.
David met Carol at the conference.
""",
        "question": "Whose car did Carol use to drive to Denver?",
        "expected_elements": ["Alice"]
    },
    {
        "id": "multihop_003",
        "name": "Numerical Chain",
        "context": """
The base price of the product is $100.
Members get a 20% discount on the base price.
Premium members get an additional 10% off the discounted price.
Tax is 8% of the final price.
Shipping is free for orders over $50.
""",
        "question": "What is the final price for a premium member including tax?",
        "expected_elements": ["77.76", "78"]  # Either exact or rounded
    }
]

# =============================================================================
# CONVERSATION MEMORY TASKS
# =============================================================================

MEMORY_CONVERSATIONS = [
    {
        "id": "memory_001",
        "name": "Personal Details",
        "exchanges": [
            {"user": "Hi, my name is Alex and I'm a software engineer.", "assistant": "Hello Alex! Nice to meet you."},
            {"user": "I've been working on a Python project for 3 months.", "assistant": "That's great!"},
            {"user": "The project is about natural language processing.", "assistant": "NLP is fascinating."},
            {"user": "I use PyTorch for the deep learning components.", "assistant": "PyTorch is excellent for that."},
        ],
        "test_questions": [
            {"question": "What's my name and profession?", "expected": ["Alex", "software engineer"]},
            {"question": "What framework am I using?", "expected": ["PyTorch"]},
            {"question": "What is my project about?", "expected": ["natural language processing", "NLP"]}
        ]
    },
    {
        "id": "memory_002",
        "name": "Task Context",
        "exchanges": [
            {"user": "I need to deploy my app to AWS.", "assistant": "I can help with AWS deployment."},
            {"user": "I'm using Docker containers.", "assistant": "Docker is great for portability."},
            {"user": "The database is PostgreSQL on RDS.", "assistant": "RDS is reliable for PostgreSQL."},
            {"user": "We need auto-scaling for traffic spikes.", "assistant": "Auto-scaling is important."},
        ],
        "test_questions": [
            {"question": "What cloud provider am I using?", "expected": ["AWS"]},
            {"question": "What database am I using?", "expected": ["PostgreSQL", "RDS"]},
            {"question": "What containerization technology did I mention?", "expected": ["Docker"]}
        ]
    }
]

# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def call_llm(messages: List[Dict], max_tokens: int = 300) -> Dict[str, Any]:
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
            timeout=180
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


def run_needle_benchmark() -> Dict[str, Any]:
    """Run needle-in-haystack benchmark"""
    print("\n  Running Needle-in-Haystack benchmark...")

    results = []

    for task in NEEDLE_TASKS:
        print(f"    [{task['id']}] {task['name']} (position: {task['position']*100:.0f}%)...")

        # Generate haystack with needle
        haystack = generate_haystack(task["needle"], task["position"], total_sentences=60)

        messages = [
            {"role": "system", "content": "Read the following text carefully and answer the question based only on the information provided."},
            {"role": "user", "content": f"Text:\n{haystack}\n\nQuestion: {task['question']}"}
        ]

        response = call_llm(messages)

        if response["success"]:
            answer = response["content"]
            found = task["expected"].lower() in answer.lower()

            result = {
                "task_id": task["id"],
                "name": task["name"],
                "success": True,
                "needle_found": found,
                "position": task["position"],
                "context_length": len(haystack.split()),
                "latency": response["latency"],
                "answer_preview": answer[:100]
            }
        else:
            result = {
                "task_id": task["id"],
                "success": False,
                "needle_found": False
            }

        results.append(result)
        status = "FOUND" if result.get("needle_found") else "MISSED"
        print(f"      {status} (latency: {result.get('latency', 0):.2f}s)")

    successful = [r for r in results if r["success"]]
    found_rate = sum(1 for r in successful if r["needle_found"]) / len(successful) if successful else 0

    # Position breakdown
    position_results = {}
    for r in successful:
        pos_bucket = "beginning" if r["position"] < 0.33 else ("middle" if r["position"] < 0.67 else "end")
        if pos_bucket not in position_results:
            position_results[pos_bucket] = {"found": 0, "total": 0}
        position_results[pos_bucket]["total"] += 1
        if r["needle_found"]:
            position_results[pos_bucket]["found"] += 1

    summary = {
        "benchmark": "Needle-in-Haystack",
        "total_tasks": len(NEEDLE_TASKS),
        "successful": len(successful),
        "found_rate": found_rate,
        "position_breakdown": {k: v["found"]/v["total"] for k, v in position_results.items()},
        "avg_latency": sum(r.get("latency", 0) for r in results) / len(results)
    }

    return {"summary": summary, "results": results}


def run_multihop_benchmark() -> Dict[str, Any]:
    """Run multi-hop context reasoning benchmark"""
    print("\n  Running Multi-hop Context benchmark...")

    results = []

    for task in MULTI_HOP_CONTEXTS:
        print(f"    [{task['id']}] {task['name']}...")

        messages = [
            {"role": "system", "content": "Carefully read the context and answer questions based on the information provided. You may need to combine multiple pieces of information."},
            {"role": "user", "content": f"Context:\n{task['context']}\n\nQuestion: {task['question']}"}
        ]

        response = call_llm(messages)

        if response["success"]:
            answer = response["content"].lower()
            elements_found = sum(1 for elem in task["expected_elements"] if elem.lower() in answer)
            element_coverage = elements_found / len(task["expected_elements"])

            result = {
                "task_id": task["id"],
                "name": task["name"],
                "success": True,
                "elements_found": elements_found,
                "total_elements": len(task["expected_elements"]),
                "element_coverage": element_coverage,
                "latency": response["latency"],
                "answer_preview": response["content"][:150]
            }
        else:
            result = {
                "task_id": task["id"],
                "success": False,
                "element_coverage": 0
            }

        results.append(result)
        print(f"      Elements: {result.get('elements_found', 0)}/{result.get('total_elements', 0)}")

    successful = [r for r in results if r["success"]]
    avg_coverage = sum(r["element_coverage"] for r in successful) / len(successful) if successful else 0

    summary = {
        "benchmark": "Multi-hop Context",
        "total_tasks": len(MULTI_HOP_CONTEXTS),
        "successful": len(successful),
        "avg_element_coverage": avg_coverage,
        "avg_latency": sum(r.get("latency", 0) for r in results) / len(results)
    }

    return {"summary": summary, "results": results}


def run_memory_benchmark() -> Dict[str, Any]:
    """Run conversation memory benchmark"""
    print("\n  Running Conversation Memory benchmark...")

    results = []

    for task in MEMORY_CONVERSATIONS:
        print(f"    [{task['id']}] {task['name']}...")

        # Build conversation history
        messages = [{"role": "system", "content": "You are a helpful assistant with good memory of our conversation."}]

        for exchange in task["exchanges"]:
            messages.append({"role": "user", "content": exchange["user"]})
            messages.append({"role": "assistant", "content": exchange["assistant"]})

        # Test memory with questions
        question_results = []
        total_latency = 0

        for test in task["test_questions"]:
            test_messages = messages.copy()
            test_messages.append({"role": "user", "content": test["question"]})

            response = call_llm(test_messages)

            if response["success"]:
                answer = response["content"].lower()
                recalled = any(exp.lower() in answer for exp in test["expected"])
                total_latency += response["latency"]

                question_results.append({
                    "question": test["question"],
                    "recalled": recalled,
                    "expected": test["expected"],
                    "answer_preview": response["content"][:100]
                })
            else:
                question_results.append({
                    "question": test["question"],
                    "recalled": False
                })

        recall_rate = sum(1 for q in question_results if q["recalled"]) / len(question_results)

        result = {
            "task_id": task["id"],
            "name": task["name"],
            "success": True,
            "question_results": question_results,
            "recall_rate": recall_rate,
            "conversation_turns": len(task["exchanges"]),
            "total_latency": total_latency
        }
        results.append(result)

        print(f"      Recall: {recall_rate*100:.0f}%")

    avg_recall = sum(r["recall_rate"] for r in results) / len(results)

    summary = {
        "benchmark": "Conversation Memory",
        "total_conversations": len(MEMORY_CONVERSATIONS),
        "avg_recall_rate": avg_recall,
        "avg_latency": sum(r["total_latency"] for r in results) / len(results)
    }

    return {"summary": summary, "results": results}


def main():
    """Run Track 09 benchmarks"""
    print("=" * 70)
    print("  ARBM TRACK 09: LONG-CONTEXT & MEMORY BENCHMARKS")
    print("  Model: Llama-3-8B-Instruct via vLLM")
    print("=" * 70)

    needle_results = run_needle_benchmark()
    multihop_results = run_multihop_benchmark()
    memory_results = run_memory_benchmark()

    # Print summary
    print("\n" + "=" * 70)
    print("  TRACK 09 SUMMARY")
    print("=" * 70)

    print("\n  Needle-in-Haystack:")
    s = needle_results["summary"]
    print(f"    Found Rate:        {s['found_rate']*100:.1f}%")
    if s.get("position_breakdown"):
        for pos, rate in s["position_breakdown"].items():
            print(f"    - {pos.capitalize()}: {rate*100:.1f}%")

    print("\n  Multi-hop Context:")
    s = multihop_results["summary"]
    print(f"    Element Coverage:  {s['avg_element_coverage']*100:.1f}%")

    print("\n  Conversation Memory:")
    s = memory_results["summary"]
    print(f"    Recall Rate:       {s['avg_recall_rate']*100:.1f}%")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = {
        "timestamp": datetime.now().isoformat(),
        "track": "09_long_context",
        "model": "llama-3-8b-instruct",
        "benchmarks": {
            "needle_in_haystack": needle_results,
            "multihop_context": multihop_results,
            "conversation_memory": memory_results
        }
    }

    outfile = f"{OUTPUT_DIR}/track_09_{timestamp}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved: {outfile}")
    print("=" * 70)


if __name__ == "__main__":
    main()
