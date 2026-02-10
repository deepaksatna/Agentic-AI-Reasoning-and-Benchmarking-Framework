#!/usr/bin/env python3
"""
ARBM Track 03: Multi-Agent Coordination Benchmarks
Evaluates Hierarchical, Collaborative, and Competitive agent patterns

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
OUTPUT_DIR = "/mnt/fss/ARBM/tracks/track_03_multi_agent/results"

# =============================================================================
# HIERARCHICAL COORDINATION TASKS
# =============================================================================

HIERARCHICAL_TASKS = [
    {
        "id": "hier_001",
        "name": "Research Report Delegation",
        "scenario": """You are an Orchestrator Agent managing a team:
- Researcher Agent: Gathers information
- Analyst Agent: Processes and analyzes data
- Writer Agent: Creates reports
- Reviewer Agent: Quality checks

Task: Create a report on "AI in Healthcare"

Describe how you would delegate tasks, what each agent should do, and in what order.""",
        "expected_elements": ["researcher", "analyst", "writer", "reviewer", "delegate", "order"],
        "evaluation": "delegation_quality"
    },
    {
        "id": "hier_002",
        "name": "Code Review Pipeline",
        "scenario": """You are a Manager Agent coordinating:
- Security Agent: Checks for vulnerabilities
- Performance Agent: Analyzes efficiency
- Style Agent: Checks code style
- Test Agent: Verifies test coverage

Code to review:
```python
def get_user(id):
    query = f"SELECT * FROM users WHERE id = {id}"
    return db.execute(query)
```

How would you coordinate these agents to review this code?""",
        "expected_elements": ["security", "sql injection", "performance", "delegate"],
        "evaluation": "delegation_quality"
    },
    {
        "id": "hier_003",
        "name": "Project Planning",
        "scenario": """You are a Project Manager Agent with:
- Design Agent: Creates system designs
- Backend Agent: Implements server logic
- Frontend Agent: Builds UI
- DevOps Agent: Handles deployment

Task: Build a user authentication system

Create a delegation plan with dependencies and timeline.""",
        "expected_elements": ["design", "backend", "frontend", "deploy", "depend"],
        "evaluation": "delegation_quality"
    }
]

# =============================================================================
# COLLABORATIVE COORDINATION TASKS
# =============================================================================

COLLABORATIVE_TASKS = [
    {
        "id": "collab_001",
        "name": "Market Entry Strategy",
        "scenario": """Three analysts must collaborate:
- Market Analyst: Understands market trends
- Financial Analyst: Evaluates costs and ROI
- Competitive Analyst: Studies competitors

They need to create a unified recommendation for entering the electric vehicle market.

Simulate the collaboration: What would each analyst contribute? How would they synthesize their findings?""",
        "expected_elements": ["market", "financial", "competitive", "consensus", "recommend"],
        "evaluation": "collaboration_quality"
    },
    {
        "id": "collab_002",
        "name": "Remote Work Debate",
        "scenario": """Three agents must debate and reach consensus:
- Pro Agent: Advocates for fully remote work
- Con Agent: Argues against remote work
- Synthesizer Agent: Combines perspectives

Generate the debate and final recommendation.""",
        "expected_elements": ["benefit", "challenge", "balance", "hybrid", "recommend"],
        "evaluation": "collaboration_quality"
    },
    {
        "id": "collab_003",
        "name": "Bug Investigation",
        "scenario": """Three debugging agents collaborate:
- Frontend Agent: Checks UI issues
- Backend Agent: Examines server logs
- Database Agent: Reviews query performance

Bug report: "User login is slow and sometimes fails"

How would they collaborate to diagnose the issue?""",
        "expected_elements": ["frontend", "backend", "database", "investigate", "root cause"],
        "evaluation": "collaboration_quality"
    }
]

# =============================================================================
# COMPETITIVE COORDINATION TASKS
# =============================================================================

COMPETITIVE_TASKS = [
    {
        "id": "comp_001",
        "name": "Solution Competition",
        "scenario": """Three agents compete to propose the best solution:
- Agent A: Proposes solution approach A
- Agent B: Proposes solution approach B
- Agent C: Proposes solution approach C
- Judge Agent: Evaluates and selects winner

Problem: Reduce a company's carbon footprint by 30%

Generate three different proposals and evaluate them.""",
        "expected_elements": ["proposal", "evaluate", "criteria", "winner", "reason"],
        "evaluation": "competition_quality"
    },
    {
        "id": "comp_002",
        "name": "Marketing Campaign",
        "scenario": """Three creative agents compete:
- Creative 1: Traditional advertising approach
- Creative 2: Social media viral campaign
- Creative 3: Influencer partnership strategy

Product: New sustainable water bottle

Judge should evaluate on: creativity, feasibility, and expected ROI.""",
        "expected_elements": ["traditional", "social", "influencer", "evaluate", "winner"],
        "evaluation": "competition_quality"
    },
    {
        "id": "comp_003",
        "name": "Algorithm Selection",
        "scenario": """Three ML agents propose different approaches:
- Agent 1: Proposes Random Forest
- Agent 2: Proposes Neural Network
- Agent 3: Proposes Gradient Boosting

Problem: Predict customer churn with 100k rows, 50 features

Evaluate each approach on accuracy, speed, and interpretability.""",
        "expected_elements": ["random forest", "neural", "gradient", "accuracy", "recommend"],
        "evaluation": "competition_quality"
    }
]

# =============================================================================
# MESSAGE PASSING TASKS
# =============================================================================

MESSAGE_PASSING_TASKS = [
    {
        "id": "msg_001",
        "name": "Information Relay",
        "scenario": """Agent A discovers: "The server is running out of memory"
Agent A must inform Agent B (System Admin) who must inform Agent C (DevOps)

Show the message chain and how information should be preserved and enhanced at each step.""",
        "expected_elements": ["memory", "inform", "action", "escalate"],
        "evaluation": "message_quality"
    },
    {
        "id": "msg_002",
        "name": "Context Aggregation",
        "scenario": """Three agents report to a coordinator:
- Agent 1: "Sales increased 15% this quarter"
- Agent 2: "Customer satisfaction dropped 5%"
- Agent 3: "Marketing spend increased 20%"

As the coordinator, synthesize these messages into a coherent summary with insights.""",
        "expected_elements": ["sales", "satisfaction", "marketing", "correlation", "insight"],
        "evaluation": "message_quality"
    }
]

# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def call_llm(prompt: str, max_tokens: int = 1500) -> Dict[str, Any]:
    """Call the LLM and return response with metrics."""
    try:
        start_time = time.time()
        response = requests.post(
            VLLM_URL,
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.3
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


def evaluate_elements(response: str, expected: List[str]) -> float:
    """Calculate what percentage of expected elements are present."""
    response_lower = response.lower()
    found = sum(1 for elem in expected if elem.lower() in response_lower)
    return found / len(expected)


def count_agents_mentioned(response: str) -> int:
    """Count distinct agents mentioned in response."""
    agent_patterns = [r'agent\s*[a-z0-9]', r'agent\s*\d+', r'\bagent\b',
                      r'researcher', r'analyst', r'writer', r'reviewer',
                      r'orchestrator', r'manager', r'coordinator']
    agents = set()
    for pattern in agent_patterns:
        matches = re.findall(pattern, response.lower())
        agents.update(matches)
    return len(agents)


def run_hierarchical_benchmark() -> Dict[str, Any]:
    """Run hierarchical coordination benchmark."""
    print("\n" + "=" * 70)
    print("  TRACK 03.1: HIERARCHICAL COORDINATION BENCHMARKS")
    print("=" * 70)

    results = []

    for task in HIERARCHICAL_TASKS:
        task_id = task["id"]
        print(f"\n[{task_id}] {task['name']}...")

        response = call_llm(task["scenario"])

        if response["success"]:
            element_coverage = evaluate_elements(response["content"], task["expected_elements"])
            agents_count = count_agents_mentioned(response["content"])

            print(f"  Element Coverage: {element_coverage*100:.0f}% | Agents: {agents_count}")

            result = {
                "task_id": task_id,
                "name": task["name"],
                "success": True,
                "element_coverage": element_coverage,
                "agents_mentioned": agents_count,
                "latency": response["latency"],
                "tokens_in": response["tokens_in"],
                "tokens_out": response["tokens_out"]
            }
        else:
            print(f"  ERROR: {response['error']}")
            result = {"task_id": task_id, "success": False, "error": response["error"]}

        results.append(result)

    successful = [r for r in results if r.get("success", False)]

    summary = {
        "benchmark": "Hierarchical Coordination",
        "total_tasks": len(HIERARCHICAL_TASKS),
        "successful": len(successful),
        "avg_element_coverage": sum(r.get("element_coverage", 0) for r in successful) / len(successful) if successful else 0,
        "avg_agents": sum(r.get("agents_mentioned", 0) for r in successful) / len(successful) if successful else 0,
        "avg_latency": sum(r["latency"] for r in successful) / len(successful) if successful else 0
    }

    return {"summary": summary, "results": results}


def run_collaborative_benchmark() -> Dict[str, Any]:
    """Run collaborative coordination benchmark."""
    print("\n" + "=" * 70)
    print("  TRACK 03.2: COLLABORATIVE COORDINATION BENCHMARKS")
    print("=" * 70)

    results = []

    for task in COLLABORATIVE_TASKS:
        task_id = task["id"]
        print(f"\n[{task_id}] {task['name']}...")

        response = call_llm(task["scenario"])

        if response["success"]:
            element_coverage = evaluate_elements(response["content"], task["expected_elements"])

            # Check for consensus indicators
            consensus_words = ["agree", "consensus", "conclusion", "together", "unified", "combined"]
            has_consensus = any(w in response["content"].lower() for w in consensus_words)

            print(f"  Coverage: {element_coverage*100:.0f}% | Consensus: {has_consensus}")

            result = {
                "task_id": task_id,
                "name": task["name"],
                "success": True,
                "element_coverage": element_coverage,
                "reached_consensus": has_consensus,
                "latency": response["latency"],
                "tokens_in": response["tokens_in"],
                "tokens_out": response["tokens_out"]
            }
        else:
            print(f"  ERROR: {response['error']}")
            result = {"task_id": task_id, "success": False, "error": response["error"]}

        results.append(result)

    successful = [r for r in results if r.get("success", False)]
    consensus_rate = sum(1 for r in successful if r.get("reached_consensus", False)) / len(successful) if successful else 0

    summary = {
        "benchmark": "Collaborative Coordination",
        "total_tasks": len(COLLABORATIVE_TASKS),
        "successful": len(successful),
        "avg_element_coverage": sum(r.get("element_coverage", 0) for r in successful) / len(successful) if successful else 0,
        "consensus_rate": consensus_rate,
        "avg_latency": sum(r["latency"] for r in successful) / len(successful) if successful else 0
    }

    return {"summary": summary, "results": results}


def run_competitive_benchmark() -> Dict[str, Any]:
    """Run competitive coordination benchmark."""
    print("\n" + "=" * 70)
    print("  TRACK 03.3: COMPETITIVE COORDINATION BENCHMARKS")
    print("=" * 70)

    results = []

    for task in COMPETITIVE_TASKS:
        task_id = task["id"]
        print(f"\n[{task_id}] {task['name']}...")

        response = call_llm(task["scenario"])

        if response["success"]:
            element_coverage = evaluate_elements(response["content"], task["expected_elements"])

            # Check for evaluation and selection
            eval_words = ["winner", "best", "recommend", "select", "choose", "evaluate"]
            has_selection = any(w in response["content"].lower() for w in eval_words)

            print(f"  Coverage: {element_coverage*100:.0f}% | Selection Made: {has_selection}")

            result = {
                "task_id": task_id,
                "name": task["name"],
                "success": True,
                "element_coverage": element_coverage,
                "made_selection": has_selection,
                "latency": response["latency"],
                "tokens_in": response["tokens_in"],
                "tokens_out": response["tokens_out"]
            }
        else:
            print(f"  ERROR: {response['error']}")
            result = {"task_id": task_id, "success": False, "error": response["error"]}

        results.append(result)

    successful = [r for r in results if r.get("success", False)]
    selection_rate = sum(1 for r in successful if r.get("made_selection", False)) / len(successful) if successful else 0

    summary = {
        "benchmark": "Competitive Coordination",
        "total_tasks": len(COMPETITIVE_TASKS),
        "successful": len(successful),
        "avg_element_coverage": sum(r.get("element_coverage", 0) for r in successful) / len(successful) if successful else 0,
        "selection_rate": selection_rate,
        "avg_latency": sum(r["latency"] for r in successful) / len(successful) if successful else 0
    }

    return {"summary": summary, "results": results}


def run_message_passing_benchmark() -> Dict[str, Any]:
    """Run message passing benchmark."""
    print("\n" + "=" * 70)
    print("  TRACK 03.4: MESSAGE PASSING BENCHMARKS")
    print("=" * 70)

    results = []

    for task in MESSAGE_PASSING_TASKS:
        task_id = task["id"]
        print(f"\n[{task_id}] {task['name']}...")

        response = call_llm(task["scenario"])

        if response["success"]:
            element_coverage = evaluate_elements(response["content"], task["expected_elements"])

            print(f"  Coverage: {element_coverage*100:.0f}%")

            result = {
                "task_id": task_id,
                "name": task["name"],
                "success": True,
                "element_coverage": element_coverage,
                "latency": response["latency"],
                "tokens_in": response["tokens_in"],
                "tokens_out": response["tokens_out"]
            }
        else:
            print(f"  ERROR: {response['error']}")
            result = {"task_id": task_id, "success": False, "error": response["error"]}

        results.append(result)

    successful = [r for r in results if r.get("success", False)]

    summary = {
        "benchmark": "Message Passing",
        "total_tasks": len(MESSAGE_PASSING_TASKS),
        "successful": len(successful),
        "avg_element_coverage": sum(r.get("element_coverage", 0) for r in successful) / len(successful) if successful else 0,
        "avg_latency": sum(r["latency"] for r in successful) / len(successful) if successful else 0
    }

    return {"summary": summary, "results": results}


def main():
    """Run all Track 03 benchmarks."""
    print("=" * 70)
    print("  ARBM TRACK 03: MULTI-AGENT COORDINATION BENCHMARKS")
    print("  Model: Llama-3-8B-Instruct via vLLM")
    print("=" * 70)

    # Run all benchmarks
    hier_results = run_hierarchical_benchmark()
    collab_results = run_collaborative_benchmark()
    comp_results = run_competitive_benchmark()
    msg_results = run_message_passing_benchmark()

    # Print summary
    print("\n" + "=" * 70)
    print("  TRACK 03 SUMMARY")
    print("=" * 70)

    for name, results in [
        ("Hierarchical", hier_results),
        ("Collaborative", collab_results),
        ("Competitive", comp_results),
        ("Message Passing", msg_results)
    ]:
        s = results["summary"]
        print(f"\n  {s['benchmark']}:")
        print(f"    Tasks: {s['total_tasks']} | Successful: {s['successful']}")
        print(f"    Element Coverage: {s['avg_element_coverage']*100:.1f}%")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = {
        "timestamp": datetime.now().isoformat(),
        "track": "03_multi_agent",
        "model": "llama-3-8b-instruct",
        "benchmarks": {
            "hierarchical": hier_results,
            "collaborative": collab_results,
            "competitive": comp_results,
            "message_passing": msg_results
        }
    }

    outfile = f"{OUTPUT_DIR}/track_03_{timestamp}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved: {outfile}")
    print("=" * 70)


if __name__ == "__main__":
    main()
