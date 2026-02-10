#!/usr/bin/env python3
"""
ARBM Track 10: Planning & Task Decomposition Benchmarks
Evaluates ability to break down complex tasks and create executable plans

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
OUTPUT_DIR = "/mnt/fss/ARBM/benchmarks/trending/track_10_planning/results"

# =============================================================================
# TASK DECOMPOSITION TASKS
# =============================================================================

DECOMPOSITION_TASKS = [
    {
        "id": "decomp_001",
        "name": "Web App Development",
        "category": "software",
        "task": "Build a user authentication system with login, signup, and password reset functionality.",
        "expected_subtasks": ["database", "API", "frontend", "email", "security", "test"],
        "min_steps": 5,
        "max_steps": 15
    },
    {
        "id": "decomp_002",
        "name": "Data Pipeline",
        "category": "data",
        "task": "Create an ETL pipeline that extracts data from multiple APIs, transforms it, and loads into a data warehouse.",
        "expected_subtasks": ["extract", "API", "transform", "clean", "load", "warehouse", "schedule"],
        "min_steps": 4,
        "max_steps": 12
    },
    {
        "id": "decomp_003",
        "name": "ML Model Deployment",
        "category": "mlops",
        "task": "Deploy a machine learning model to production with monitoring and auto-scaling.",
        "expected_subtasks": ["model", "container", "API", "scale", "monitor", "log"],
        "min_steps": 5,
        "max_steps": 12
    },
    {
        "id": "decomp_004",
        "name": "E-commerce Feature",
        "category": "product",
        "task": "Implement a shopping cart feature with add/remove items, quantity updates, and checkout flow.",
        "expected_subtasks": ["cart", "add", "remove", "quantity", "checkout", "payment", "order"],
        "min_steps": 5,
        "max_steps": 15
    },
    {
        "id": "decomp_005",
        "name": "Security Audit",
        "category": "security",
        "task": "Perform a security audit on a web application and implement necessary fixes.",
        "expected_subtasks": ["scan", "vulnerability", "authentication", "authorization", "input", "encrypt", "fix"],
        "min_steps": 5,
        "max_steps": 12
    }
]

# =============================================================================
# DEPENDENCY RESOLUTION TASKS
# =============================================================================

DEPENDENCY_TASKS = [
    {
        "id": "depend_001",
        "name": "Build Order",
        "category": "dependencies",
        "scenario": """
You have the following tasks with dependencies:
- Task A: No dependencies
- Task B: Depends on A
- Task C: Depends on A
- Task D: Depends on B and C
- Task E: Depends on D
""",
        "question": "What is a valid order to execute these tasks?",
        "valid_orders": [
            ["A", "B", "C", "D", "E"],
            ["A", "C", "B", "D", "E"]
        ],
        "must_be_before": [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E")]
    },
    {
        "id": "depend_002",
        "name": "Circular Detection",
        "category": "dependencies",
        "scenario": """
You have the following tasks:
- Task X: Depends on Z
- Task Y: Depends on X
- Task Z: Depends on Y
""",
        "question": "Can these tasks be executed? If not, why?",
        "expected_detection": "circular",
        "keywords": ["circular", "cycle", "impossible", "cannot", "loop"]
    },
    {
        "id": "depend_003",
        "name": "Parallel Execution",
        "category": "optimization",
        "scenario": """
Tasks and their dependencies:
- A: No dependencies (2 mins)
- B: No dependencies (3 mins)
- C: Depends on A (1 min)
- D: Depends on B (2 mins)
- E: Depends on C and D (1 min)
""",
        "question": "What tasks can run in parallel? What is the minimum total time?",
        "parallel_groups": [["A", "B"]],
        "min_time": 6,  # max(2+1, 3+2) + 1 = 5+1 = 6
        "keywords": ["parallel", "concurrent", "same time", "6", "minutes"]
    }
]

# =============================================================================
# RESOURCE ALLOCATION TASKS
# =============================================================================

RESOURCE_TASKS = [
    {
        "id": "resource_001",
        "name": "Team Assignment",
        "category": "allocation",
        "scenario": """
You have 3 developers (Alice, Bob, Charlie) and 5 tasks:
- Task 1: Frontend (needs JavaScript skills) - Alice has JS, Bob has Python, Charlie has both
- Task 2: Backend API (needs Python)
- Task 3: Database (needs SQL) - All have SQL
- Task 4: Testing (anyone can do)
- Task 5: Documentation (anyone can do)

Each person can work on one task at a time.
""",
        "question": "Propose an optimal assignment to complete all tasks quickly.",
        "constraints": ["Alice", "Bob", "Charlie"],
        "optimization_keywords": ["parallel", "skills", "assign", "efficient"]
    },
    {
        "id": "resource_002",
        "name": "Budget Allocation",
        "category": "allocation",
        "scenario": """
You have a $10,000 budget for a software project with these needs:
- Cloud hosting: Essential, $2000-5000 range
- Development tools: Essential, $500-1000 range
- Testing services: Important, $1000-2000 range
- Documentation tools: Nice to have, $200-500 range
- Training: Nice to have, $1000-2000 range
""",
        "question": "How would you allocate the budget to maximize value?",
        "must_include": ["cloud", "development", "testing"],
        "total_budget": 10000
    }
]

# =============================================================================
# GOAL-ORIENTED PLANNING TASKS
# =============================================================================

GOAL_TASKS = [
    {
        "id": "goal_001",
        "name": "Startup Launch",
        "category": "strategic",
        "goal": "Launch an MVP of a mobile app in 3 months with a team of 3.",
        "constraints": ["3 months", "3 people", "MVP"],
        "expected_phases": ["design", "develop", "test", "launch"],
        "expected_elements": ["scope", "timeline", "milestones", "risk"]
    },
    {
        "id": "goal_002",
        "name": "System Migration",
        "category": "technical",
        "goal": "Migrate a legacy monolith application to microservices architecture with zero downtime.",
        "constraints": ["zero downtime", "microservices"],
        "expected_phases": ["analyze", "design", "migrate", "test", "cutover"],
        "expected_elements": ["strangler", "incremental", "rollback", "monitoring"]
    },
    {
        "id": "goal_003",
        "name": "Performance Optimization",
        "category": "technical",
        "goal": "Improve API response time from 2 seconds to under 200ms.",
        "constraints": ["200ms target", "API"],
        "expected_phases": ["profile", "identify", "optimize", "measure"],
        "expected_elements": ["cache", "database", "query", "index", "benchmark"]
    }
]

# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def call_llm(prompt: str, system: str = None, max_tokens: int = 800) -> Dict[str, Any]:
    """Call the LLM"""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

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
                "latency": latency
            }
        return {"success": False, "error": response.text[:200], "latency": latency}
    except Exception as e:
        return {"success": False, "error": str(e)[:200], "latency": 0}


def count_steps(text: str) -> int:
    """Count numbered steps in text"""
    patterns = [
        r'^\d+[\.\)]\s',  # 1. or 1)
        r'^[-•]\s',  # bullet points
        r'^Step \d+',  # Step 1
    ]
    lines = text.split('\n')
    count = 0
    for line in lines:
        for pattern in patterns:
            if re.match(pattern, line.strip()):
                count += 1
                break
    return max(count, 1)  # At least 1 if there's content


def run_decomposition_benchmark() -> Dict[str, Any]:
    """Run task decomposition benchmark"""
    print("\n  Running Task Decomposition benchmark...")

    results = []
    system = "You are a project manager expert at breaking down complex tasks into actionable steps. Provide numbered steps."

    for task in DECOMPOSITION_TASKS:
        print(f"    [{task['id']}] {task['name']}...")

        prompt = f"Break down this task into detailed steps:\n\n{task['task']}\n\nProvide a numbered list of steps."

        response = call_llm(prompt, system)

        if response["success"]:
            content = response["content"].lower()
            steps = count_steps(response["content"])

            # Check for expected subtasks
            subtasks_found = sum(1 for st in task["expected_subtasks"] if st.lower() in content)
            subtask_coverage = subtasks_found / len(task["expected_subtasks"])

            # Check step count quality
            step_quality = 1.0 if task["min_steps"] <= steps <= task["max_steps"] else 0.5

            result = {
                "task_id": task["id"],
                "name": task["name"],
                "category": task["category"],
                "success": True,
                "steps_generated": steps,
                "subtask_coverage": subtask_coverage,
                "step_quality": step_quality,
                "combined_score": (subtask_coverage + step_quality) / 2,
                "latency": response["latency"],
                "plan_preview": response["content"][:300]
            }
        else:
            result = {
                "task_id": task["id"],
                "success": False,
                "combined_score": 0
            }

        results.append(result)
        print(f"      Steps: {result.get('steps_generated', 0)}, Coverage: {result.get('subtask_coverage', 0)*100:.0f}%")

    successful = [r for r in results if r["success"]]
    avg_score = sum(r["combined_score"] for r in successful) / len(successful) if successful else 0

    summary = {
        "benchmark": "Task Decomposition",
        "total_tasks": len(DECOMPOSITION_TASKS),
        "successful": len(successful),
        "avg_combined_score": avg_score,
        "avg_steps": sum(r.get("steps_generated", 0) for r in successful) / len(successful) if successful else 0,
        "avg_subtask_coverage": sum(r.get("subtask_coverage", 0) for r in successful) / len(successful) if successful else 0
    }

    return {"summary": summary, "results": results}


def run_dependency_benchmark() -> Dict[str, Any]:
    """Run dependency resolution benchmark"""
    print("\n  Running Dependency Resolution benchmark...")

    results = []
    system = "You are an expert at analyzing task dependencies and execution order."

    for task in DEPENDENCY_TASKS:
        print(f"    [{task['id']}] {task['name']}...")

        prompt = f"{task['scenario']}\n\nQuestion: {task['question']}"

        response = call_llm(prompt, system)

        if response["success"]:
            content = response["content"]

            if "expected_detection" in task:
                # Circular dependency detection
                detected = any(kw in content.lower() for kw in task["keywords"])
                result = {
                    "task_id": task["id"],
                    "name": task["name"],
                    "category": task["category"],
                    "success": True,
                    "detection_correct": detected,
                    "score": 1.0 if detected else 0.0,
                    "latency": response["latency"],
                    "response_preview": content[:200]
                }
            elif "must_be_before" in task:
                # Order validation
                # Extract mentioned task order from response
                order_correct = True
                for before, after in task["must_be_before"]:
                    before_pos = content.find(before)
                    after_pos = content.find(after)
                    if before_pos == -1 or after_pos == -1 or before_pos > after_pos:
                        # Check if both are mentioned in correct logical relationship
                        if f"{before}" not in content or f"{after}" not in content:
                            order_correct = False
                            break

                result = {
                    "task_id": task["id"],
                    "name": task["name"],
                    "category": task["category"],
                    "success": True,
                    "order_correct": order_correct,
                    "score": 1.0 if order_correct else 0.5,
                    "latency": response["latency"],
                    "response_preview": content[:200]
                }
            else:
                # Parallel execution detection
                keywords_found = sum(1 for kw in task["keywords"] if kw.lower() in content.lower())
                result = {
                    "task_id": task["id"],
                    "name": task["name"],
                    "category": task["category"],
                    "success": True,
                    "keywords_found": keywords_found,
                    "score": keywords_found / len(task["keywords"]),
                    "latency": response["latency"],
                    "response_preview": content[:200]
                }
        else:
            result = {
                "task_id": task["id"],
                "success": False,
                "score": 0
            }

        results.append(result)
        print(f"      Score: {result.get('score', 0)*100:.0f}%")

    successful = [r for r in results if r["success"]]
    avg_score = sum(r["score"] for r in successful) / len(successful) if successful else 0

    summary = {
        "benchmark": "Dependency Resolution",
        "total_tasks": len(DEPENDENCY_TASKS),
        "successful": len(successful),
        "avg_score": avg_score
    }

    return {"summary": summary, "results": results}


def run_goal_planning_benchmark() -> Dict[str, Any]:
    """Run goal-oriented planning benchmark"""
    print("\n  Running Goal-Oriented Planning benchmark...")

    results = []
    system = "You are a strategic planner expert at creating comprehensive project plans."

    for task in GOAL_TASKS:
        print(f"    [{task['id']}] {task['name']}...")

        prompt = f"""Create a comprehensive plan to achieve this goal:

Goal: {task['goal']}

Constraints: {', '.join(task['constraints'])}

Provide:
1. Key phases with milestones
2. Risk considerations
3. Success metrics"""

        response = call_llm(prompt, system, max_tokens=1000)

        if response["success"]:
            content = response["content"].lower()

            # Check phases
            phases_found = sum(1 for phase in task["expected_phases"] if phase in content)
            phase_coverage = phases_found / len(task["expected_phases"])

            # Check elements
            elements_found = sum(1 for elem in task["expected_elements"] if elem in content)
            element_coverage = elements_found / len(task["expected_elements"])

            # Check constraints mentioned
            constraints_addressed = sum(1 for c in task["constraints"] if c.lower() in content)
            constraint_coverage = constraints_addressed / len(task["constraints"])

            combined_score = (phase_coverage + element_coverage + constraint_coverage) / 3

            result = {
                "task_id": task["id"],
                "name": task["name"],
                "category": task["category"],
                "success": True,
                "phase_coverage": phase_coverage,
                "element_coverage": element_coverage,
                "constraint_coverage": constraint_coverage,
                "combined_score": combined_score,
                "latency": response["latency"],
                "plan_preview": response["content"][:400]
            }
        else:
            result = {
                "task_id": task["id"],
                "success": False,
                "combined_score": 0
            }

        results.append(result)
        print(f"      Score: {result.get('combined_score', 0)*100:.0f}%")

    successful = [r for r in results if r["success"]]
    avg_score = sum(r["combined_score"] for r in successful) / len(successful) if successful else 0

    summary = {
        "benchmark": "Goal-Oriented Planning",
        "total_tasks": len(GOAL_TASKS),
        "successful": len(successful),
        "avg_combined_score": avg_score,
        "avg_phase_coverage": sum(r.get("phase_coverage", 0) for r in successful) / len(successful) if successful else 0,
        "avg_element_coverage": sum(r.get("element_coverage", 0) for r in successful) / len(successful) if successful else 0
    }

    return {"summary": summary, "results": results}


def main():
    """Run Track 10 benchmarks"""
    print("=" * 70)
    print("  ARBM TRACK 10: PLANNING & TASK DECOMPOSITION BENCHMARKS")
    print("  Model: Llama-3-8B-Instruct via vLLM")
    print("=" * 70)

    decomp_results = run_decomposition_benchmark()
    depend_results = run_dependency_benchmark()
    goal_results = run_goal_planning_benchmark()

    # Print summary
    print("\n" + "=" * 70)
    print("  TRACK 10 SUMMARY")
    print("=" * 70)

    print("\n  Task Decomposition:")
    s = decomp_results["summary"]
    print(f"    Avg Score:           {s['avg_combined_score']*100:.1f}%")
    print(f"    Avg Steps:           {s['avg_steps']:.1f}")
    print(f"    Subtask Coverage:    {s['avg_subtask_coverage']*100:.1f}%")

    print("\n  Dependency Resolution:")
    s = depend_results["summary"]
    print(f"    Avg Score:           {s['avg_score']*100:.1f}%")

    print("\n  Goal-Oriented Planning:")
    s = goal_results["summary"]
    print(f"    Avg Score:           {s['avg_combined_score']*100:.1f}%")
    print(f"    Phase Coverage:      {s['avg_phase_coverage']*100:.1f}%")
    print(f"    Element Coverage:    {s['avg_element_coverage']*100:.1f}%")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = {
        "timestamp": datetime.now().isoformat(),
        "track": "10_planning",
        "model": "llama-3-8b-instruct",
        "benchmarks": {
            "task_decomposition": decomp_results,
            "dependency_resolution": depend_results,
            "goal_planning": goal_results
        }
    }

    outfile = f"{OUTPUT_DIR}/track_10_{timestamp}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved: {outfile}")
    print("=" * 70)


if __name__ == "__main__":
    main()
