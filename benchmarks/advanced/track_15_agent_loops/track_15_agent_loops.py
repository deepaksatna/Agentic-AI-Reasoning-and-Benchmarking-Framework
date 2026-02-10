#!/usr/bin/env python3
"""
ARBM Track 15: Agent Loops - Extended Multi-Turn Sessions
Evaluates ability to maintain context and perform in extended agentic workflows

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
OUTPUT_DIR = "/mnt/fss/ARBM/benchmarks/advanced/track_15_agent_loops/results"

# Multi-Turn Task Completion
TASK_SESSIONS = [
    {
        "id": "session_001",
        "name": "Research Assistant",
        "system": "You are a research assistant helping with a project.",
        "turns": [
            {"user": "I'm researching renewable energy. What are the main types?", "check_keywords": ["solar", "wind", "hydro"]},
            {"user": "Focus on solar energy. What are the key technologies?", "check_keywords": ["photovoltaic", "panel", "cell"]},
            {"user": "What are the main challenges with solar adoption?", "check_keywords": ["cost", "storage", "weather", "efficiency"]},
            {"user": "Based on our discussion, summarize the solar energy landscape in 2 sentences.", "check_keywords": ["solar", "energy"]}
        ]
    },
    {
        "id": "session_002",
        "name": "Code Development",
        "system": "You are a programming assistant helping build a Python application.",
        "turns": [
            {"user": "I need a function to validate email addresses. What should it check?", "check_keywords": ["@", "domain", "format"]},
            {"user": "Write the basic function with regex.", "check_keywords": ["def", "import re", "pattern", "match"]},
            {"user": "Add error handling for None and empty strings.", "check_keywords": ["if", "None", "empty", "return"]},
            {"user": "Now write a test case for this function.", "check_keywords": ["test", "assert", "valid", "invalid"]}
        ]
    },
    {
        "id": "session_003",
        "name": "Project Planning",
        "system": "You are a project manager helping plan a software project.",
        "turns": [
            {"user": "We're building a mobile app for task management. What phases do we need?", "check_keywords": ["design", "development", "testing", "launch"]},
            {"user": "For the design phase, what deliverables should we have?", "check_keywords": ["wireframe", "mockup", "user", "interface"]},
            {"user": "Who should be involved in the design phase?", "check_keywords": ["designer", "product", "stakeholder"]},
            {"user": "Create a simple timeline for the design phase with 3 milestones.", "check_keywords": ["week", "milestone", "review", "complete"]}
        ]
    }
]

# Context Retention Tests
CONTEXT_TESTS = [
    {
        "id": "context_001",
        "name": "Personal Info Retention",
        "turns": [
            {"user": "My name is Alexandra and I'm a data scientist from Boston.", "info": {"name": "Alexandra", "job": "data scientist", "city": "Boston"}},
            {"user": "I've been working with Python for 7 years.", "info": {"experience": "7 years", "language": "Python"}},
            {"user": "What's my name and profession?", "expected": ["Alexandra", "data scientist"]},
            {"user": "How long have I been using Python?", "expected": ["7", "seven"]}
        ]
    },
    {
        "id": "context_002",
        "name": "Task State Retention",
        "turns": [
            {"user": "I'm planning a trip to Japan in April.", "info": {"destination": "Japan", "month": "April"}},
            {"user": "I want to visit Tokyo, Kyoto, and Osaka.", "info": {"cities": ["Tokyo", "Kyoto", "Osaka"]}},
            {"user": "My budget is $3000 for 10 days.", "info": {"budget": "$3000", "duration": "10 days"}},
            {"user": "Summarize my trip plans including destination, cities, budget, and duration.", "expected": ["Japan", "Tokyo", "Kyoto", "Osaka", "3000", "10"]}
        ]
    }
]

# Error Recovery Tests
ERROR_RECOVERY_TESTS = [
    {
        "id": "recovery_001",
        "name": "Correction Handling",
        "turns": [
            {"user": "The meeting is on Tuesday at 3pm.", "type": "info"},
            {"user": "Actually, I made a mistake. The meeting is on Wednesday, not Tuesday.", "type": "correction"},
            {"user": "When is the meeting?", "expected": ["Wednesday"], "not_expected": ["Tuesday"]}
        ]
    },
    {
        "id": "recovery_002",
        "name": "Clarification Request",
        "turns": [
            {"user": "Calculate the area.", "type": "ambiguous"},
            {"user": "Sorry, I meant the area of a circle with radius 5.", "type": "clarification"},
            {"user": "What's the answer?", "expected": ["78", "79", "25*pi", "25π"]}
        ]
    }
]

# Instruction Persistence Tests
PERSISTENCE_TESTS = [
    {
        "id": "persist_001",
        "name": "Format Persistence",
        "system": "Always respond in bullet points.",
        "turns": [
            {"user": "What are 3 benefits of exercise?", "check_format": lambda x: x.count("•") >= 3 or x.count("-") >= 3 or x.count("*") >= 3},
            {"user": "Now tell me 3 benefits of reading.", "check_format": lambda x: x.count("•") >= 3 or x.count("-") >= 3 or x.count("*") >= 3},
            {"user": "And 3 benefits of meditation.", "check_format": lambda x: x.count("•") >= 3 or x.count("-") >= 3 or x.count("*") >= 3}
        ]
    },
    {
        "id": "persist_002",
        "name": "Role Persistence",
        "system": "You are a friendly pirate. Use pirate expressions in every response.",
        "turns": [
            {"user": "How's the weather?", "check_role": ["arr", "matey", "ahoy", "ye", "aye"]},
            {"user": "What should I eat for lunch?", "check_role": ["arr", "matey", "ahoy", "ye", "aye", "grub", "feast"]},
            {"user": "Tell me a joke.", "check_role": ["arr", "matey", "ahoy", "ye", "aye"]}
        ]
    }
]


def call_llm(messages: List[Dict], max_tokens: int = 400) -> Dict[str, Any]:
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


def run_task_sessions_benchmark() -> Dict[str, Any]:
    print("\n  Running Multi-Turn Task Sessions...")
    results = []

    for session in TASK_SESSIONS:
        print(f"    [{session['id']}] {session['name']}...")
        messages = [{"role": "system", "content": session["system"]}]
        turn_results = []
        total_latency = 0

        for i, turn in enumerate(session["turns"]):
            messages.append({"role": "user", "content": turn["user"]})
            response = call_llm(messages)

            if response["success"]:
                content = response["content"].lower()
                keywords_found = sum(1 for kw in turn["check_keywords"] if kw.lower() in content)
                keyword_score = keywords_found / len(turn["check_keywords"])
                total_latency += response["latency"]

                turn_results.append({
                    "turn": i + 1,
                    "keyword_score": keyword_score,
                    "latency": response["latency"]
                })
                messages.append({"role": "assistant", "content": response["content"]})
            else:
                turn_results.append({"turn": i + 1, "keyword_score": 0, "error": True})

        avg_score = sum(t["keyword_score"] for t in turn_results) / len(turn_results)
        result = {
            "session_id": session["id"],
            "name": session["name"],
            "success": True,
            "turns": turn_results,
            "avg_keyword_score": avg_score,
            "total_turns": len(session["turns"]),
            "total_latency": total_latency
        }
        results.append(result)
        print(f"      Avg Score: {avg_score*100:.0f}%")

    avg_session_score = sum(r["avg_keyword_score"] for r in results) / len(results)
    return {
        "summary": {"benchmark": "Task Sessions", "total": len(TASK_SESSIONS), "avg_score": avg_session_score},
        "results": results
    }


def run_context_retention_benchmark() -> Dict[str, Any]:
    print("\n  Running Context Retention Tests...")
    results = []

    for test in CONTEXT_TESTS:
        print(f"    [{test['id']}] {test['name']}...")
        messages = [{"role": "system", "content": "You are a helpful assistant with good memory."}]
        test_results = []

        for turn in test["turns"]:
            messages.append({"role": "user", "content": turn["user"]})
            response = call_llm(messages)

            if response["success"]:
                messages.append({"role": "assistant", "content": response["content"]})

                if "expected" in turn:
                    content = response["content"].lower()
                    found = sum(1 for exp in turn["expected"] if exp.lower() in content)
                    recall_score = found / len(turn["expected"])
                    test_results.append({"type": "recall", "score": recall_score})

        if test_results:
            avg_recall = sum(t["score"] for t in test_results) / len(test_results)
        else:
            avg_recall = 0

        result = {
            "test_id": test["id"],
            "name": test["name"],
            "success": True,
            "recall_score": avg_recall
        }
        results.append(result)
        print(f"      Recall: {avg_recall*100:.0f}%")

    avg_recall = sum(r["recall_score"] for r in results) / len(results)
    return {
        "summary": {"benchmark": "Context Retention", "total": len(CONTEXT_TESTS), "avg_recall": avg_recall},
        "results": results
    }


def run_error_recovery_benchmark() -> Dict[str, Any]:
    print("\n  Running Error Recovery Tests...")
    results = []

    for test in ERROR_RECOVERY_TESTS:
        print(f"    [{test['id']}] {test['name']}...")
        messages = [{"role": "system", "content": "You are a helpful assistant that adapts to corrections."}]

        for turn in test["turns"]:
            messages.append({"role": "user", "content": turn["user"]})
            response = call_llm(messages)

            if response["success"]:
                messages.append({"role": "assistant", "content": response["content"]})

                if "expected" in turn:
                    content = response["content"].lower()
                    found_expected = any(exp.lower() in content for exp in turn["expected"])
                    found_not_expected = any(ne.lower() in content for ne in turn.get("not_expected", []))
                    recovery_success = found_expected and not found_not_expected

                    result = {
                        "test_id": test["id"],
                        "name": test["name"],
                        "success": True,
                        "recovery_success": recovery_success
                    }
                    results.append(result)
                    status = "RECOVERED" if recovery_success else "FAILED"
                    print(f"      {status}")

    recovered = sum(1 for r in results if r.get("recovery_success"))
    return {
        "summary": {"benchmark": "Error Recovery", "total": len(results), "recovery_rate": recovered / len(results) if results else 0},
        "results": results
    }


def run_persistence_benchmark() -> Dict[str, Any]:
    print("\n  Running Instruction Persistence Tests...")
    results = []

    for test in PERSISTENCE_TESTS:
        print(f"    [{test['id']}] {test['name']}...")
        messages = [{"role": "system", "content": test["system"]}]
        turn_results = []

        for i, turn in enumerate(test["turns"]):
            messages.append({"role": "user", "content": turn["user"]})
            response = call_llm(messages)

            if response["success"]:
                content = response["content"]
                messages.append({"role": "assistant", "content": content})

                if "check_format" in turn:
                    passed = turn["check_format"](content)
                    turn_results.append({"turn": i + 1, "passed": passed, "type": "format"})
                elif "check_role" in turn:
                    passed = any(indicator in content.lower() for indicator in turn["check_role"])
                    turn_results.append({"turn": i + 1, "passed": passed, "type": "role"})

        persistence_rate = sum(1 for t in turn_results if t["passed"]) / len(turn_results) if turn_results else 0
        result = {
            "test_id": test["id"],
            "name": test["name"],
            "success": True,
            "turn_results": turn_results,
            "persistence_rate": persistence_rate
        }
        results.append(result)
        print(f"      Persistence: {persistence_rate*100:.0f}%")

    avg_persistence = sum(r["persistence_rate"] for r in results) / len(results)
    return {
        "summary": {"benchmark": "Instruction Persistence", "total": len(PERSISTENCE_TESTS), "avg_persistence": avg_persistence},
        "results": results
    }


def main():
    print("=" * 70)
    print("  ARBM TRACK 15: AGENT LOOPS - EXTENDED SESSIONS")
    print("  Model: Llama-3-8B-Instruct via vLLM")
    print("=" * 70)

    task_results = run_task_sessions_benchmark()
    context_results = run_context_retention_benchmark()
    recovery_results = run_error_recovery_benchmark()
    persistence_results = run_persistence_benchmark()

    print("\n" + "=" * 70)
    print("  TRACK 15 SUMMARY")
    print("=" * 70)
    print(f"\n  Task Sessions:          {task_results['summary']['avg_score']*100:.1f}%")
    print(f"  Context Retention:      {context_results['summary']['avg_recall']*100:.1f}%")
    print(f"  Error Recovery:         {recovery_results['summary']['recovery_rate']*100:.1f}%")
    print(f"  Instruction Persistence: {persistence_results['summary']['avg_persistence']*100:.1f}%")

    overall = (task_results['summary']['avg_score'] + context_results['summary']['avg_recall'] +
               recovery_results['summary']['recovery_rate'] + persistence_results['summary']['avg_persistence']) / 4
    print(f"\n  Overall Agent Loop Score: {overall*100:.1f}%")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "track": "15_agent_loops",
        "model": "llama-3-8b-instruct",
        "benchmarks": {
            "task_sessions": task_results,
            "context_retention": context_results,
            "error_recovery": recovery_results,
            "instruction_persistence": persistence_results
        },
        "overall_score": overall
    }

    outfile = f"{OUTPUT_DIR}/track_15_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved: {outfile}")
    print("=" * 70)


if __name__ == "__main__":
    main()
