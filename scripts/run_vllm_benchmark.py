#!/usr/bin/env python3
import requests
import json
import time
from datetime import datetime
import os

print("=" * 60)
print("  ARBM Reasoning Benchmark - vLLM")
print("=" * 60)

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "/mnt/fss/2026-NIM-vLLM_LLM/models/llama-3-8b-instruct"

TASKS = [
    {"id": "cot_math_001", "name": "Multi-step Math",
     "q": "A bakery sells cupcakes for $3 each. If they make 48 cupcakes and sell 75% of them, how much revenue do they generate? Think step by step.",
     "expected": "108"},
    {"id": "cot_logic_001", "name": "Logical Deduction",
     "q": "If all programmers use computers, and some computer users are gamers, can we conclude that some programmers are gamers? Answer yes or no with explanation.",
     "expected": "no"},
    {"id": "cot_reasoning_001", "name": "Multi-hop Reasoning",
     "q": "Alice is taller than Bob. Bob is taller than Charlie. David is shorter than Charlie. Who is the tallest?",
     "expected": "Alice"},
    {"id": "code_gen_001", "name": "Code Generation",
     "q": "Write a Python function to check if a number is prime. Just the function code.",
     "expected": "def"},
    {"id": "creative_001", "name": "Creative Problem",
     "q": "You have 3 boxes labeled Apples, Oranges, and Mixed. All labels are wrong. You can pick one fruit from one box. How do you correctly label all boxes?",
     "expected": "Mixed"}
]

results = []
total_latency = 0
successful = 0

for task in TASKS:
    task_id = task["id"]
    task_name = task["name"]
    print(f"\n[{task_id}] {task_name}...")

    try:
        start = time.time()
        resp = requests.post(VLLM_URL, json={
            "model": MODEL,
            "messages": [{"role": "user", "content": task["q"]}],
            "max_tokens": 512,
            "temperature": 0.1
        }, timeout=120)
        latency = time.time() - start

        if resp.status_code == 200:
            data = resp.json()
            answer = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {})
            correct = task["expected"].lower() in answer.lower()

            print(f"  Latency: {latency:.2f}s | Correct: {correct}")
            preview = answer[:100].replace("\n", " ")
            print(f"  Answer preview: {preview}...")

            results.append({
                "task_id": task_id,
                "success": True,
                "correct": correct,
                "latency": latency,
                "tokens_in": tokens.get("prompt_tokens", 0),
                "tokens_out": tokens.get("completion_tokens", 0)
            })
            successful += 1
            total_latency += latency
        else:
            err = resp.text[:100]
            print(f"  ERROR: {err}")
            results.append({"task_id": task_id, "success": False, "error": resp.text[:200]})
    except Exception as e:
        err = str(e)[:100]
        print(f"  EXCEPTION: {err}")
        results.append({"task_id": task_id, "success": False, "error": str(e)[:200]})

print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  Tasks: {len(TASKS)}")
print(f"  Successful: {successful}")
pct = 100*successful/len(TASKS)
print(f"  Success Rate: {pct:.1f}%")
if successful > 0:
    correct_count = sum(1 for r in results if r.get("correct", False))
    print(f"  Correct Answers: {correct_count}/{successful}")
    avg_lat = total_latency/successful
    print(f"  Avg Latency: {avg_lat:.2f}s")

os.makedirs("/mnt/fss/ARBM/benchmarks/results", exist_ok=True)
output = {
    "timestamp": datetime.now().isoformat(),
    "model": "llama-3-8b-instruct",
    "provider": "vllm_local",
    "summary": {
        "total": len(TASKS),
        "successful": successful,
        "success_rate": successful/len(TASKS),
        "avg_latency": total_latency/successful if successful > 0 else 0
    },
    "results": results
}
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
outfile = "/mnt/fss/ARBM/benchmarks/results/arbm_llama_" + timestamp_str + ".json"
with open(outfile, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved: {outfile}")
