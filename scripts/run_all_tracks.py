#!/usr/bin/env python3
"""
ARBM - Run All Benchmark Tracks
Master script to execute all 5 benchmark tracks sequentially

Author: Deepak Soni
License: MIT
"""

import subprocess
import sys
import os
import json
from datetime import datetime

SCRIPT_DIR = "/mnt/fss/ARBM/scripts"
RESULTS_DIR = "/mnt/fss/ARBM/benchmarks/results"

def run_track(track_num: int, track_name: str, script_name: str) -> bool:
    """Run a single track and return success status."""
    print("\n" + "=" * 80)
    print(f"  RUNNING TRACK {track_num:02d}: {track_name}")
    print("=" * 80 + "\n")

    script_path = f"{SCRIPT_DIR}/{script_name}"

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=SCRIPT_DIR,
            timeout=1800  # 30 minute timeout per track
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  ERROR: Track {track_num} timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"  ERROR: Failed to run track {track_num}: {e}")
        return False


def main():
    """Run all benchmark tracks."""
    print("=" * 80)
    print("  ARBM - REASONING AGENT BENCHMARK FRAMEWORK")
    print("  Running All Benchmark Tracks")
    print("=" * 80)
    print(f"  Start Time: {datetime.now().isoformat()}")
    print(f"  Model: Llama-3-8B-Instruct via vLLM")
    print("=" * 80)

    tracks = [
        (1, "Reasoning Quality (CoT, ToT, GoT)", "track_01_reasoning.py"),
        (2, "Tool-use Efficiency", "track_02_tool_use.py"),
        (3, "Multi-Agent Coordination", "track_03_multi_agent.py"),
        (4, "Safety and Observability", "track_04_safety.py"),
        (5, "Production Metrics", "track_05_production.py"),
    ]

    results = {}
    successful_tracks = 0

    for track_num, track_name, script_name in tracks:
        success = run_track(track_num, track_name, script_name)
        results[f"track_{track_num:02d}"] = {
            "name": track_name,
            "success": success
        }
        if success:
            successful_tracks += 1

    # Print final summary
    print("\n" + "=" * 80)
    print("  ARBM BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"  End Time: {datetime.now().isoformat()}")
    print(f"  Tracks Run: {len(tracks)}")
    print(f"  Successful: {successful_tracks}/{len(tracks)}")
    print("\n  Track Results:")
    for track_id, info in results.items():
        status = "PASS" if info["success"] else "FAIL"
        print(f"    {track_id}: {info['name']} - {status}")

    # Save overall summary
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_tracks": len(tracks),
        "successful_tracks": successful_tracks,
        "tracks": results
    }

    summary_file = f"{RESULTS_DIR}/arbm_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary saved: {summary_file}")
    print("=" * 80)

    return 0 if successful_tracks == len(tracks) else 1


if __name__ == "__main__":
    sys.exit(main())
