#!/usr/bin/env python3
"""
ARBM - Agent Benchmarks Core Module
Implements benchmarks for agentic AI workflows with reasoning models

Author: Deepak Soni
License: MIT
"""

import os
import sys
import json
import time
import yaml
import logging
import requests
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# Data Classes
# ============================================================

@dataclass
class BenchmarkResult:
    """Result from a single benchmark run"""
    task_id: str
    task_name: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    success: bool
    score: float
    reasoning_trace: List[str]
    tool_calls: List[Dict]
    error: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class TrackResult:
    """Aggregated results for a benchmark track"""
    track_id: str
    track_name: str
    timestamp: str
    total_tasks: int
    completed_tasks: int
    success_rate: float
    avg_score: float
    avg_latency_ms: float
    total_tokens: int
    results: List[BenchmarkResult]


class ReasoningType(Enum):
    """Types of reasoning approaches"""
    CHAIN_OF_THOUGHT = "cot"
    TREE_OF_THOUGHTS = "tot"
    GRAPH_OF_THOUGHTS = "got"


# ============================================================
# Base Agent Class
# ============================================================

class BaseAgent(ABC):
    """Abstract base class for AI agents"""

    def __init__(self, model: str, provider: str, api_key: Optional[str] = None):
        self.model = model
        self.provider = provider
        self.api_key = api_key or os.environ.get(f"{provider.upper()}_API_KEY")
        self.reasoning_trace = []
        self.tool_calls = []

    @abstractmethod
    def chat(self, messages: List[Dict], max_tokens: int = 1000,
             temperature: float = 0.0) -> Tuple[str, Dict]:
        """Send a chat completion request"""
        pass

    @abstractmethod
    def call_tool(self, tool_name: str, parameters: Dict) -> Dict:
        """Execute a tool call"""
        pass

    def reset(self):
        """Reset agent state"""
        self.reasoning_trace = []
        self.tool_calls = []


# ============================================================
# Provider Implementations
# ============================================================

class AnthropicAgent(BaseAgent):
    """Anthropic Claude agent implementation"""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022",
                 api_key: Optional[str] = None):
        super().__init__(model, "anthropic", api_key)
        self.api_base = "https://api.anthropic.com/v1"

    def chat(self, messages: List[Dict], max_tokens: int = 1000,
             temperature: float = 0.0) -> Tuple[str, Dict]:
        """Send chat completion to Claude API"""
        headers = {
            "x-api-key": self.api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01"
        }

        # Convert messages format
        system_msg = None
        claude_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                claude_messages.append(msg)

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": claude_messages
        }
        if system_msg:
            payload["system"] = system_msg

        start_time = time.time()
        try:
            response = requests.post(
                f"{self.api_base}/messages",
                headers=headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            latency_ms = (time.time() - start_time) * 1000

            content = result["content"][0]["text"]
            metadata = {
                "input_tokens": result["usage"]["input_tokens"],
                "output_tokens": result["usage"]["output_tokens"],
                "latency_ms": latency_ms,
                "stop_reason": result["stop_reason"]
            }
            return content, metadata

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    def call_tool(self, tool_name: str, parameters: Dict) -> Dict:
        """Execute tool call via Claude"""
        # Tool use implementation
        pass


class OpenAIAgent(BaseAgent):
    """OpenAI GPT agent implementation"""

    def __init__(self, model: str = "gpt-4-turbo",
                 api_key: Optional[str] = None):
        super().__init__(model, "openai", api_key)
        self.api_base = "https://api.openai.com/v1"

    def chat(self, messages: List[Dict], max_tokens: int = 1000,
             temperature: float = 0.0) -> Tuple[str, Dict]:
        """Send chat completion to OpenAI API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        start_time = time.time()
        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            latency_ms = (time.time() - start_time) * 1000

            content = result["choices"][0]["message"]["content"]
            metadata = {
                "input_tokens": result["usage"]["prompt_tokens"],
                "output_tokens": result["usage"]["completion_tokens"],
                "latency_ms": latency_ms,
                "finish_reason": result["choices"][0]["finish_reason"]
            }
            return content, metadata

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def call_tool(self, tool_name: str, parameters: Dict) -> Dict:
        """Execute tool call via GPT function calling"""
        pass


class VLLMAgent(BaseAgent):
    """vLLM local inference agent"""

    def __init__(self, model: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                 endpoint: str = "http://localhost:8000"):
        super().__init__(model, "vllm", None)
        self.endpoint = endpoint

    def chat(self, messages: List[Dict], max_tokens: int = 1000,
             temperature: float = 0.0) -> Tuple[str, Dict]:
        """Send chat completion to vLLM server"""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        start_time = time.time()
        try:
            response = requests.post(
                f"{self.endpoint}/v1/chat/completions",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            latency_ms = (time.time() - start_time) * 1000

            content = result["choices"][0]["message"]["content"]
            metadata = {
                "input_tokens": result.get("usage", {}).get("prompt_tokens", 0),
                "output_tokens": result.get("usage", {}).get("completion_tokens", 0),
                "latency_ms": latency_ms
            }
            return content, metadata

        except Exception as e:
            logger.error(f"vLLM API error: {e}")
            raise

    def call_tool(self, tool_name: str, parameters: Dict) -> Dict:
        pass


# ============================================================
# Reasoning Benchmarks
# ============================================================

class ReasoningBenchmark:
    """Benchmark for reasoning quality evaluation"""

    def __init__(self, agent: BaseAgent, config: Dict):
        self.agent = agent
        self.config = config
        self.results = []

    def run_cot_benchmark(self, task: Dict) -> BenchmarkResult:
        """Run Chain-of-Thought reasoning benchmark"""
        prompt = f"""Solve this problem step by step. Show your reasoning clearly.

Problem: {task['question']}

Think through this step by step:"""

        messages = [{"role": "user", "content": prompt}]

        start_time = time.time()
        try:
            response, metadata = self.agent.chat(messages, max_tokens=1000)
            latency_ms = (time.time() - start_time) * 1000

            # Extract reasoning steps
            steps = self._extract_reasoning_steps(response)

            # Evaluate answer
            score = self._evaluate_cot_response(response, task)

            return BenchmarkResult(
                task_id=task.get("id", "unknown"),
                task_name=task.get("name", "CoT Benchmark"),
                model=self.agent.model,
                provider=self.agent.provider,
                input_tokens=metadata.get("input_tokens", 0),
                output_tokens=metadata.get("output_tokens", 0),
                latency_ms=latency_ms,
                success=score > 0.5,
                score=score,
                reasoning_trace=steps,
                tool_calls=[],
                metadata={"response": response[:500]}
            )

        except Exception as e:
            return BenchmarkResult(
                task_id=task.get("id", "unknown"),
                task_name=task.get("name", "CoT Benchmark"),
                model=self.agent.model,
                provider=self.agent.provider,
                input_tokens=0,
                output_tokens=0,
                latency_ms=0,
                success=False,
                score=0.0,
                reasoning_trace=[],
                tool_calls=[],
                error=str(e)
            )

    def run_tot_benchmark(self, task: Dict) -> BenchmarkResult:
        """Run Tree of Thoughts reasoning benchmark"""
        prompt = f"""Solve this problem by exploring multiple approaches.

Problem: {task['question']}

Consider different approaches:
1. First, brainstorm 3 different ways to solve this
2. Evaluate each approach
3. Select the best one and solve

Show your exploration:"""

        messages = [{"role": "user", "content": prompt}]

        start_time = time.time()
        try:
            response, metadata = self.agent.chat(messages, max_tokens=1500)
            latency_ms = (time.time() - start_time) * 1000

            # Extract branches explored
            branches = self._extract_tot_branches(response)

            # Evaluate answer
            score = self._evaluate_tot_response(response, task)

            return BenchmarkResult(
                task_id=task.get("id", "unknown"),
                task_name=task.get("name", "ToT Benchmark"),
                model=self.agent.model,
                provider=self.agent.provider,
                input_tokens=metadata.get("input_tokens", 0),
                output_tokens=metadata.get("output_tokens", 0),
                latency_ms=latency_ms,
                success=score > 0.5,
                score=score,
                reasoning_trace=branches,
                tool_calls=[],
                metadata={"response": response[:500], "branches_explored": len(branches)}
            )

        except Exception as e:
            return BenchmarkResult(
                task_id=task.get("id", "unknown"),
                task_name=task.get("name", "ToT Benchmark"),
                model=self.agent.model,
                provider=self.agent.provider,
                input_tokens=0,
                output_tokens=0,
                latency_ms=0,
                success=False,
                score=0.0,
                reasoning_trace=[],
                tool_calls=[],
                error=str(e)
            )

    def _extract_reasoning_steps(self, response: str) -> List[str]:
        """Extract reasoning steps from response"""
        steps = []
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or
                        line.lower().startswith('step')):
                steps.append(line)
        return steps if steps else [response[:200]]

    def _extract_tot_branches(self, response: str) -> List[str]:
        """Extract ToT branches from response"""
        branches = []
        keywords = ['approach', 'method', 'way', 'option', 'alternative']
        lines = response.split('\n')
        for line in lines:
            if any(kw in line.lower() for kw in keywords):
                branches.append(line.strip())
        return branches if branches else ["single_branch"]

    def _evaluate_cot_response(self, response: str, task: Dict) -> float:
        """Evaluate CoT response quality"""
        score = 0.0

        # Check if answer is present
        if task.get("answer"):
            if str(task["answer"]).lower() in response.lower():
                score += 0.5

        # Check for reasoning steps
        steps = self._extract_reasoning_steps(response)
        if len(steps) >= 2:
            score += 0.3
        if len(steps) >= 4:
            score += 0.2

        return min(score, 1.0)

    def _evaluate_tot_response(self, response: str, task: Dict) -> float:
        """Evaluate ToT response quality"""
        score = 0.0

        # Check for multiple approaches
        branches = self._extract_tot_branches(response)
        if len(branches) >= 2:
            score += 0.3
        if len(branches) >= 3:
            score += 0.2

        # Check for evaluation
        if 'best' in response.lower() or 'select' in response.lower():
            score += 0.2

        # Check answer
        if task.get("answer"):
            if str(task["answer"]).lower() in response.lower():
                score += 0.3

        return min(score, 1.0)


# ============================================================
# Tool Use Benchmarks
# ============================================================

class ToolUseBenchmark:
    """Benchmark for tool use efficiency"""

    def __init__(self, agent: BaseAgent, config: Dict):
        self.agent = agent
        self.config = config
        self.available_tools = self._setup_tools()

    def _setup_tools(self) -> Dict[str, callable]:
        """Setup available tools for benchmarking"""
        return {
            "web_search": self._mock_web_search,
            "code_execute": self._mock_code_execute,
            "memory_retrieve": self._mock_memory_retrieve,
        }

    def _mock_web_search(self, query: str) -> Dict:
        """Mock web search tool"""
        return {
            "results": [
                {"title": f"Result for: {query}", "snippet": "Mock search result..."}
            ],
            "latency_ms": 150
        }

    def _mock_code_execute(self, code: str) -> Dict:
        """Mock code execution tool"""
        return {
            "output": "Code executed successfully",
            "error": None,
            "latency_ms": 50
        }

    def _mock_memory_retrieve(self, query: str) -> Dict:
        """Mock memory retrieval tool"""
        return {
            "memories": ["Retrieved memory context..."],
            "latency_ms": 20
        }

    def run_tool_benchmark(self, task: Dict) -> BenchmarkResult:
        """Run tool use benchmark"""
        prompt = f"""You have access to the following tools:
- web_search(query): Search the web for information
- code_execute(code): Execute Python code
- memory_retrieve(query): Retrieve from memory

Task: {task['instruction']}

Think about which tools to use and in what order. Then explain your approach."""

        messages = [{"role": "user", "content": prompt}]

        start_time = time.time()
        try:
            response, metadata = self.agent.chat(messages, max_tokens=1000)
            latency_ms = (time.time() - start_time) * 1000

            # Analyze tool selection
            tool_calls = self._analyze_tool_selection(response, task)
            score = self._evaluate_tool_use(response, task, tool_calls)

            return BenchmarkResult(
                task_id=task.get("id", "unknown"),
                task_name=task.get("name", "Tool Use Benchmark"),
                model=self.agent.model,
                provider=self.agent.provider,
                input_tokens=metadata.get("input_tokens", 0),
                output_tokens=metadata.get("output_tokens", 0),
                latency_ms=latency_ms,
                success=score > 0.5,
                score=score,
                reasoning_trace=[response[:300]],
                tool_calls=tool_calls,
                metadata={"expected_tools": task.get("expected_tools", [])}
            )

        except Exception as e:
            return BenchmarkResult(
                task_id=task.get("id", "unknown"),
                task_name=task.get("name", "Tool Use Benchmark"),
                model=self.agent.model,
                provider=self.agent.provider,
                input_tokens=0,
                output_tokens=0,
                latency_ms=0,
                success=False,
                score=0.0,
                reasoning_trace=[],
                tool_calls=[],
                error=str(e)
            )

    def _analyze_tool_selection(self, response: str, task: Dict) -> List[Dict]:
        """Analyze which tools the agent mentioned/selected"""
        tool_calls = []
        for tool_name in self.available_tools.keys():
            if tool_name in response.lower():
                tool_calls.append({"tool": tool_name, "mentioned": True})
        return tool_calls

    def _evaluate_tool_use(self, response: str, task: Dict,
                          tool_calls: List[Dict]) -> float:
        """Evaluate tool use effectiveness"""
        score = 0.0
        expected_tools = task.get("expected_tools", [])

        # Check if correct tools were identified
        mentioned_tools = [tc["tool"] for tc in tool_calls]
        for expected in expected_tools:
            if expected in mentioned_tools:
                score += 0.3

        # Check for reasoning about tool use
        reasoning_keywords = ['because', 'first', 'then', 'order', 'need']
        if any(kw in response.lower() for kw in reasoning_keywords):
            score += 0.2

        return min(score, 1.0)


# ============================================================
# Multi-Agent Benchmarks
# ============================================================

class MultiAgentBenchmark:
    """Benchmark for multi-agent coordination"""

    def __init__(self, agents: List[BaseAgent], config: Dict):
        self.agents = agents
        self.config = config
        self.conversation_log = []

    def run_coordination_benchmark(self, task: Dict) -> BenchmarkResult:
        """Run multi-agent coordination benchmark"""
        # Simulate multi-agent interaction
        # In a real implementation, this would involve actual agent coordination

        orchestrator = self.agents[0] if self.agents else None
        if not orchestrator:
            return self._error_result(task, "No agents configured")

        prompt = f"""You are an orchestrator agent. Your task is to delegate and coordinate.

Task: {task['instruction']}

Available specialists: researcher, analyst, writer

Plan how to coordinate these agents to complete the task.
Output a step-by-step delegation plan."""

        messages = [{"role": "user", "content": prompt}]

        start_time = time.time()
        try:
            response, metadata = orchestrator.chat(messages, max_tokens=1200)
            latency_ms = (time.time() - start_time) * 1000

            # Analyze coordination
            delegation_steps = self._analyze_delegation(response)
            score = self._evaluate_coordination(response, task)

            return BenchmarkResult(
                task_id=task.get("id", "unknown"),
                task_name=task.get("name", "Multi-Agent Benchmark"),
                model=orchestrator.model,
                provider=orchestrator.provider,
                input_tokens=metadata.get("input_tokens", 0),
                output_tokens=metadata.get("output_tokens", 0),
                latency_ms=latency_ms,
                success=score > 0.5,
                score=score,
                reasoning_trace=delegation_steps,
                tool_calls=[],
                metadata={"agents_involved": len(self.agents)}
            )

        except Exception as e:
            return self._error_result(task, str(e))

    def _analyze_delegation(self, response: str) -> List[str]:
        """Analyze delegation steps"""
        steps = []
        agent_keywords = ['researcher', 'analyst', 'writer', 'delegate', 'assign']
        lines = response.split('\n')
        for line in lines:
            if any(kw in line.lower() for kw in agent_keywords):
                steps.append(line.strip())
        return steps

    def _evaluate_coordination(self, response: str, task: Dict) -> float:
        """Evaluate coordination quality"""
        score = 0.0

        # Check for clear delegation
        if 'delegate' in response.lower() or 'assign' in response.lower():
            score += 0.3

        # Check for multiple agents mentioned
        agents_mentioned = sum(1 for a in ['researcher', 'analyst', 'writer']
                              if a in response.lower())
        score += min(agents_mentioned * 0.2, 0.4)

        # Check for coordination logic
        if 'then' in response.lower() or 'after' in response.lower():
            score += 0.2

        return min(score, 1.0)

    def _error_result(self, task: Dict, error: str) -> BenchmarkResult:
        """Create error result"""
        return BenchmarkResult(
            task_id=task.get("id", "unknown"),
            task_name=task.get("name", "Multi-Agent Benchmark"),
            model="unknown",
            provider="unknown",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
            success=False,
            score=0.0,
            reasoning_trace=[],
            tool_calls=[],
            error=error
        )


# ============================================================
# Track Runner
# ============================================================

class TrackRunner:
    """Run benchmark tracks and aggregate results"""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.results_dir = self.config.get("output", {}).get("directory", "results")
        os.makedirs(self.results_dir, exist_ok=True)

    def run_track(self, track_id: str, tasks: List[Dict],
                  agent: BaseAgent) -> TrackResult:
        """Run a complete benchmark track"""
        logger.info(f"Running track: {track_id}")

        results = []

        if track_id == "track_01":
            benchmark = ReasoningBenchmark(agent, self.config)
            for task in tasks:
                if task.get("type") == "cot":
                    result = benchmark.run_cot_benchmark(task)
                elif task.get("type") == "tot":
                    result = benchmark.run_tot_benchmark(task)
                else:
                    result = benchmark.run_cot_benchmark(task)
                results.append(result)
                logger.info(f"  Task {task.get('id')}: score={result.score:.2f}")

        elif track_id == "track_02":
            benchmark = ToolUseBenchmark(agent, self.config)
            for task in tasks:
                result = benchmark.run_tool_benchmark(task)
                results.append(result)
                logger.info(f"  Task {task.get('id')}: score={result.score:.2f}")

        # Calculate aggregates
        completed = [r for r in results if not r.error]

        track_result = TrackResult(
            track_id=track_id,
            track_name=self.config.get("track", {}).get("name", track_id),
            timestamp=datetime.now().isoformat(),
            total_tasks=len(tasks),
            completed_tasks=len(completed),
            success_rate=sum(1 for r in completed if r.success) / len(completed) if completed else 0,
            avg_score=sum(r.score for r in completed) / len(completed) if completed else 0,
            avg_latency_ms=sum(r.latency_ms for r in completed) / len(completed) if completed else 0,
            total_tokens=sum(r.input_tokens + r.output_tokens for r in completed),
            results=results
        )

        # Save results
        self._save_results(track_result)

        return track_result

    def _save_results(self, track_result: TrackResult):
        """Save track results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.results_dir}/{track_result.track_id}_{timestamp}.json"

        # Convert to serializable format
        data = {
            "track_id": track_result.track_id,
            "track_name": track_result.track_name,
            "timestamp": track_result.timestamp,
            "summary": {
                "total_tasks": track_result.total_tasks,
                "completed_tasks": track_result.completed_tasks,
                "success_rate": track_result.success_rate,
                "avg_score": track_result.avg_score,
                "avg_latency_ms": track_result.avg_latency_ms,
                "total_tokens": track_result.total_tokens
            },
            "results": [asdict(r) for r in track_result.results]
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Results saved to: {filename}")


# ============================================================
# Main Entry Point
# ============================================================

def main():
    """Main entry point for running benchmarks"""
    print("="*60)
    print("  ARBM - Reasoning-Agent-Benchmark Framework")
    print("="*60)

    # Example usage
    sample_tasks = [
        {
            "id": "cot_001",
            "type": "cot",
            "name": "Math Reasoning",
            "question": "If a store sells apples at $2 each and oranges at $3 each, "
                       "and you buy 5 apples and 3 oranges, how much do you spend in total?",
            "answer": "19"
        },
        {
            "id": "cot_002",
            "type": "cot",
            "name": "Logic Puzzle",
            "question": "All roses are flowers. Some flowers fade quickly. "
                       "Can we conclude that some roses fade quickly?",
            "answer": "no"
        }
    ]

    print("\nTo run benchmarks, use:")
    print("  python3 run_all_tracks.py --config configs/benchmark_config.yaml")
    print("\nOr run individual tracks:")
    print("  python3 run_track.py --track 01 --config configs/track_01_config.yaml")


if __name__ == "__main__":
    main()
