#!/usr/bin/env python3
"""
ARBM Track 05: Production Metrics Benchmarks
Evaluates Latency, Throughput, and Cost metrics across tasks

Author: Deepak Soni
License: MIT
"""

import requests
import json
import time
import re
import statistics
from datetime import datetime
from typing import Dict, List, Any
import os
import concurrent.futures

# =============================================================================
# CONFIGURATION
# =============================================================================

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1/chat/completions")
MODEL = os.environ.get("MODEL_PATH", "/mnt/fss/2026-NIM-vLLM_LLM/models/llama-3-8b-instruct")
OUTPUT_DIR = "/mnt/fss/ARBM/tracks/track_05_production/results"

# =============================================================================
# BENCHMARK TASK CATEGORIES
# =============================================================================

# Short prompts for latency testing
SHORT_PROMPTS = [
    {"id": "short_001", "prompt": "What is 2 + 2?", "max_tokens": 50},
    {"id": "short_002", "prompt": "Name the capital of France.", "max_tokens": 50},
    {"id": "short_003", "prompt": "What color is the sky?", "max_tokens": 50},
    {"id": "short_004", "prompt": "Say hello in Spanish.", "max_tokens": 50},
    {"id": "short_005", "prompt": "What is Python?", "max_tokens": 100},
]

# Medium prompts for balanced testing
MEDIUM_PROMPTS = [
    {"id": "medium_001", "prompt": "Explain the concept of machine learning in simple terms.", "max_tokens": 300},
    {"id": "medium_002", "prompt": "Write a short paragraph about climate change.", "max_tokens": 300},
    {"id": "medium_003", "prompt": "Describe the process of photosynthesis.", "max_tokens": 300},
    {"id": "medium_004", "prompt": "Explain what an API is and why it's useful.", "max_tokens": 300},
    {"id": "medium_005", "prompt": "Summarize the benefits of exercise.", "max_tokens": 300},
]

# Long prompts for throughput testing
LONG_PROMPTS = [
    {"id": "long_001", "prompt": "Write a detailed essay about the impact of artificial intelligence on modern society, including benefits, challenges, and future prospects.", "max_tokens": 800},
    {"id": "long_002", "prompt": "Explain the complete process of how a web application works, from the user clicking a button to receiving a response, including all layers involved.", "max_tokens": 800},
    {"id": "long_003", "prompt": "Describe the history of computing from the 1940s to today, highlighting major milestones and their significance.", "max_tokens": 800},
]

# Complex reasoning prompts
COMPLEX_PROMPTS = [
    {"id": "complex_001", "prompt": "A company has 500 employees. 60% work in engineering, 25% in sales, and the rest in operations. If the company grows by 20% next year with the same distribution, how many engineers will there be? Show all steps.", "max_tokens": 500},
    {"id": "complex_002", "prompt": "Analyze the pros and cons of remote work vs office work, considering productivity, work-life balance, collaboration, and company culture. Provide a balanced conclusion.", "max_tokens": 600},
    {"id": "complex_003", "prompt": "Design a simple database schema for an e-commerce platform, including tables for users, products, orders, and reviews. Explain the relationships between tables.", "max_tokens": 600},
]

# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def call_llm(prompt: str, max_tokens: int = 512) -> Dict[str, Any]:
    """Call the LLM and return response with detailed metrics."""
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
            timeout=300
        )
        latency = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

            # Calculate tokens per second
            tokens_per_second = completion_tokens / latency if latency > 0 else 0

            return {
                "success": True,
                "content": data["choices"][0]["message"]["content"],
                "latency": latency,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "tokens_per_second": tokens_per_second,
                "time_to_first_token": latency / 2  # Approximation
            }
        else:
            return {"success": False, "error": response.text[:200], "latency": latency}
    except Exception as e:
        return {"success": False, "error": str(e)[:200], "latency": 0}


def run_latency_benchmark(prompts: List[Dict], category: str, iterations: int = 3) -> Dict[str, Any]:
    """Run latency benchmark for a category of prompts."""
    print(f"\n  Running {category} latency tests ({len(prompts)} prompts x {iterations} iterations)...")

    results = []

    for prompt_data in prompts:
        prompt_results = []
        for i in range(iterations):
            response = call_llm(prompt_data["prompt"], prompt_data["max_tokens"])
            if response["success"]:
                prompt_results.append({
                    "iteration": i + 1,
                    "latency": response["latency"],
                    "tokens_per_second": response["tokens_per_second"],
                    "completion_tokens": response["completion_tokens"]
                })

        if prompt_results:
            latencies = [r["latency"] for r in prompt_results]
            tps = [r["tokens_per_second"] for r in prompt_results]

            result = {
                "prompt_id": prompt_data["id"],
                "iterations": len(prompt_results),
                "avg_latency": statistics.mean(latencies),
                "min_latency": min(latencies),
                "max_latency": max(latencies),
                "std_latency": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                "avg_tokens_per_second": statistics.mean(tps),
                "avg_completion_tokens": statistics.mean([r["completion_tokens"] for r in prompt_results])
            }
            results.append(result)
            print(f"    [{prompt_data['id']}] Avg: {result['avg_latency']:.2f}s, TPS: {result['avg_tokens_per_second']:.1f}")

    # Calculate category statistics
    if results:
        all_latencies = [r["avg_latency"] for r in results]
        all_tps = [r["avg_tokens_per_second"] for r in results]

        summary = {
            "category": category,
            "total_prompts": len(prompts),
            "successful_prompts": len(results),
            "avg_latency": statistics.mean(all_latencies),
            "p50_latency": statistics.median(all_latencies),
            "p95_latency": sorted(all_latencies)[int(len(all_latencies) * 0.95)] if len(all_latencies) > 1 else all_latencies[0],
            "avg_tokens_per_second": statistics.mean(all_tps),
            "total_tokens_generated": sum(r["avg_completion_tokens"] for r in results)
        }
    else:
        summary = {"category": category, "error": "No successful results"}

    return {"summary": summary, "results": results}


def run_throughput_benchmark() -> Dict[str, Any]:
    """Run throughput benchmark with concurrent requests."""
    print("\n  Running throughput benchmark...")

    # Test with different concurrency levels
    concurrency_levels = [1, 2, 4]
    throughput_results = []

    test_prompt = "Write a short paragraph about technology."
    max_tokens = 200
    requests_per_level = 5

    for concurrency in concurrency_levels:
        print(f"    Testing concurrency level: {concurrency}")

        start_time = time.time()
        total_tokens = 0
        successful = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(call_llm, test_prompt, max_tokens)
                for _ in range(requests_per_level)
            ]

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result["success"]:
                    successful += 1
                    total_tokens += result["completion_tokens"]

        elapsed = time.time() - start_time
        requests_per_second = successful / elapsed if elapsed > 0 else 0
        tokens_per_second = total_tokens / elapsed if elapsed > 0 else 0

        throughput_results.append({
            "concurrency": concurrency,
            "requests_sent": requests_per_level,
            "successful_requests": successful,
            "elapsed_seconds": elapsed,
            "requests_per_second": requests_per_second,
            "tokens_per_second": tokens_per_second
        })

        print(f"      RPS: {requests_per_second:.2f}, TPS: {tokens_per_second:.1f}")

    # Find optimal concurrency
    best = max(throughput_results, key=lambda x: x["tokens_per_second"])

    summary = {
        "benchmark": "Throughput",
        "optimal_concurrency": best["concurrency"],
        "max_requests_per_second": best["requests_per_second"],
        "max_tokens_per_second": best["tokens_per_second"]
    }

    return {"summary": summary, "results": throughput_results}


def run_cost_estimation() -> Dict[str, Any]:
    """Estimate costs based on token usage."""
    print("\n  Running cost estimation benchmark...")

    # Run a mix of prompts
    test_prompts = [
        ("short", "What is 2+2?", 50),
        ("medium", "Explain machine learning.", 300),
        ("long", "Write an essay about AI.", 800)
    ]

    # Cost assumptions (per 1M tokens) - typical API pricing
    INPUT_COST_PER_1M = 3.00  # $3 per 1M input tokens
    OUTPUT_COST_PER_1M = 15.00  # $15 per 1M output tokens

    results = []
    total_input_tokens = 0
    total_output_tokens = 0

    for category, prompt, max_tokens in test_prompts:
        response = call_llm(prompt, max_tokens)
        if response["success"]:
            input_tokens = response["prompt_tokens"]
            output_tokens = response["completion_tokens"]

            input_cost = (input_tokens / 1_000_000) * INPUT_COST_PER_1M
            output_cost = (output_tokens / 1_000_000) * OUTPUT_COST_PER_1M
            total_cost = input_cost + output_cost

            results.append({
                "category": category,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": total_cost,
                "cost_per_token": total_cost / (input_tokens + output_tokens) if (input_tokens + output_tokens) > 0 else 0
            })

            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            print(f"    [{category}] Tokens: {input_tokens}+{output_tokens}, Cost: ${total_cost:.6f}")

    # Calculate totals
    total_input_cost = (total_input_tokens / 1_000_000) * INPUT_COST_PER_1M
    total_output_cost = (total_output_tokens / 1_000_000) * OUTPUT_COST_PER_1M

    summary = {
        "benchmark": "Cost Estimation",
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_input_cost": total_input_cost,
        "total_output_cost": total_output_cost,
        "total_cost": total_input_cost + total_output_cost,
        "cost_per_1k_tokens": ((total_input_cost + total_output_cost) / (total_input_tokens + total_output_tokens)) * 1000 if (total_input_tokens + total_output_tokens) > 0 else 0
    }

    return {"summary": summary, "results": results}


def run_reliability_benchmark() -> Dict[str, Any]:
    """Test reliability with repeated requests."""
    print("\n  Running reliability benchmark...")

    test_prompt = "What is the capital of Japan?"
    num_requests = 10

    results = []
    successes = 0
    failures = 0
    latencies = []

    for i in range(num_requests):
        response = call_llm(test_prompt, 50)
        if response["success"]:
            successes += 1
            latencies.append(response["latency"])
            results.append({"request": i + 1, "success": True, "latency": response["latency"]})
        else:
            failures += 1
            results.append({"request": i + 1, "success": False, "error": response.get("error", "Unknown")})

    success_rate = successes / num_requests
    avg_latency = statistics.mean(latencies) if latencies else 0
    latency_variance = statistics.variance(latencies) if len(latencies) > 1 else 0

    print(f"    Success Rate: {success_rate*100:.1f}%, Avg Latency: {avg_latency:.2f}s")

    summary = {
        "benchmark": "Reliability",
        "total_requests": num_requests,
        "successes": successes,
        "failures": failures,
        "success_rate": success_rate,
        "avg_latency": avg_latency,
        "latency_variance": latency_variance
    }

    return {"summary": summary, "results": results}


def main():
    """Run all Track 05 benchmarks."""
    print("=" * 70)
    print("  ARBM TRACK 05: PRODUCTION METRICS BENCHMARKS")
    print("  Model: Llama-3-8B-Instruct via vLLM")
    print("=" * 70)

    # Run latency benchmarks
    print("\n" + "-" * 70)
    print("  LATENCY BENCHMARKS")
    print("-" * 70)

    short_results = run_latency_benchmark(SHORT_PROMPTS, "short", iterations=2)
    medium_results = run_latency_benchmark(MEDIUM_PROMPTS, "medium", iterations=2)
    long_results = run_latency_benchmark(LONG_PROMPTS, "long", iterations=2)
    complex_results = run_latency_benchmark(COMPLEX_PROMPTS, "complex", iterations=2)

    # Run throughput benchmark
    print("\n" + "-" * 70)
    print("  THROUGHPUT BENCHMARK")
    print("-" * 70)
    throughput_results = run_throughput_benchmark()

    # Run cost estimation
    print("\n" + "-" * 70)
    print("  COST ESTIMATION")
    print("-" * 70)
    cost_results = run_cost_estimation()

    # Run reliability benchmark
    print("\n" + "-" * 70)
    print("  RELIABILITY BENCHMARK")
    print("-" * 70)
    reliability_results = run_reliability_benchmark()

    # Print summary
    print("\n" + "=" * 70)
    print("  TRACK 05 SUMMARY")
    print("=" * 70)

    print("\n  Latency by Category:")
    for name, results in [("Short", short_results), ("Medium", medium_results),
                          ("Long", long_results), ("Complex", complex_results)]:
        s = results["summary"]
        if "avg_latency" in s:
            print(f"    {name}: Avg {s['avg_latency']:.2f}s, P50 {s.get('p50_latency', 0):.2f}s, TPS {s.get('avg_tokens_per_second', 0):.1f}")

    print(f"\n  Throughput:")
    ts = throughput_results["summary"]
    print(f"    Optimal Concurrency: {ts['optimal_concurrency']}")
    print(f"    Max RPS: {ts['max_requests_per_second']:.2f}")
    print(f"    Max TPS: {ts['max_tokens_per_second']:.1f}")

    print(f"\n  Cost (estimated):")
    cs = cost_results["summary"]
    print(f"    Total Tokens: {cs['total_input_tokens'] + cs['total_output_tokens']}")
    print(f"    Total Cost: ${cs['total_cost']:.6f}")

    print(f"\n  Reliability:")
    rs = reliability_results["summary"]
    print(f"    Success Rate: {rs['success_rate']*100:.1f}%")
    print(f"    Avg Latency: {rs['avg_latency']:.2f}s")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = {
        "timestamp": datetime.now().isoformat(),
        "track": "05_production",
        "model": "llama-3-8b-instruct",
        "benchmarks": {
            "latency": {
                "short": short_results,
                "medium": medium_results,
                "long": long_results,
                "complex": complex_results
            },
            "throughput": throughput_results,
            "cost": cost_results,
            "reliability": reliability_results
        }
    }

    outfile = f"{OUTPUT_DIR}/track_05_{timestamp}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved: {outfile}")
    print("=" * 70)


if __name__ == "__main__":
    main()
