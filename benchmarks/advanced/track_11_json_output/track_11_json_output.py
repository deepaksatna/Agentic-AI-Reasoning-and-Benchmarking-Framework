#!/usr/bin/env python3
"""
ARBM Track 11: Structured Output & JSON Generation Benchmarks
Evaluates ability to generate valid JSON and follow output schemas

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
OUTPUT_DIR = "/mnt/fss/ARBM/benchmarks/advanced/track_11_json_output/results"

# JSON Generation Tasks
JSON_TASKS = [
    {
        "id": "json_001",
        "name": "Simple Object",
        "prompt": "Generate a JSON object representing a person with fields: name (string), age (integer), email (string). Use realistic values.",
        "required_keys": ["name", "age", "email"],
        "types": {"name": str, "age": int, "email": str}
    },
    {
        "id": "json_002",
        "name": "Nested Object",
        "prompt": "Generate a JSON object for a product with: id (integer), name (string), price (float), category (object with name and id), tags (array of strings).",
        "required_keys": ["id", "name", "price", "category", "tags"],
        "types": {"id": int, "name": str, "price": (int, float), "category": dict, "tags": list}
    },
    {
        "id": "json_003",
        "name": "Array of Objects",
        "prompt": "Generate a JSON array containing 3 book objects. Each book should have: title, author, year, isbn.",
        "is_array": True,
        "min_items": 3,
        "item_keys": ["title", "author", "year", "isbn"]
    },
    {
        "id": "json_004",
        "name": "API Response Format",
        "prompt": "Generate a JSON API response with: status (string 'success' or 'error'), data (object with items array and total count), timestamp (ISO format string).",
        "required_keys": ["status", "data", "timestamp"],
        "nested_checks": {"data": ["items", "total"]}
    },
    {
        "id": "json_005",
        "name": "Config File",
        "prompt": "Generate a JSON configuration file with: database (object with host, port, name), logging (object with level, file), features (object with boolean flags).",
        "required_keys": ["database", "logging", "features"],
        "nested_checks": {"database": ["host", "port", "name"], "logging": ["level", "file"]}
    }
]

# Schema Following Tasks
SCHEMA_TASKS = [
    {
        "id": "schema_001",
        "name": "User Schema",
        "schema": {
            "type": "object",
            "properties": {
                "username": {"type": "string", "minLength": 3},
                "email": {"type": "string", "format": "email"},
                "role": {"type": "string", "enum": ["admin", "user", "guest"]}
            },
            "required": ["username", "email", "role"]
        },
        "prompt": "Generate a valid JSON object following this schema: username (string, min 3 chars), email (valid email format), role (one of: admin, user, guest)"
    },
    {
        "id": "schema_002",
        "name": "Event Schema",
        "schema": {
            "type": "object",
            "properties": {
                "event_name": {"type": "string"},
                "date": {"type": "string", "format": "date"},
                "attendees": {"type": "integer", "minimum": 0},
                "is_virtual": {"type": "boolean"}
            },
            "required": ["event_name", "date", "attendees", "is_virtual"]
        },
        "prompt": "Generate a JSON event object with: event_name (string), date (YYYY-MM-DD format), attendees (positive integer), is_virtual (boolean)"
    },
    {
        "id": "schema_003",
        "name": "Order Schema",
        "schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "items": {"type": "array", "items": {"type": "object"}},
                "total": {"type": "number"},
                "status": {"type": "string", "enum": ["pending", "processing", "shipped", "delivered"]}
            }
        },
        "prompt": "Generate a JSON order with: order_id (string), items (array of objects with name and quantity), total (number), status (pending/processing/shipped/delivered)"
    }
]

# Data Extraction to JSON
EXTRACTION_TASKS = [
    {
        "id": "extract_001",
        "name": "Text to JSON",
        "input": "John Smith is a 35-year-old software engineer from San Francisco. He has 10 years of experience and specializes in Python and JavaScript.",
        "prompt": "Extract the following information from the text and return as JSON: name, age, occupation, location, years_experience, skills (array)",
        "expected_keys": ["name", "age", "occupation", "location", "years_experience", "skills"]
    },
    {
        "id": "extract_002",
        "name": "Receipt to JSON",
        "input": "Receipt #12345 - Date: 2024-01-15. Items: Coffee $4.50, Sandwich $8.99, Cookie $2.50. Subtotal: $15.99, Tax: $1.28, Total: $17.27",
        "prompt": "Parse this receipt into JSON with: receipt_id, date, items (array with name and price), subtotal, tax, total",
        "expected_keys": ["receipt_id", "date", "items", "subtotal", "tax", "total"]
    },
    {
        "id": "extract_003",
        "name": "Contact to JSON",
        "input": "Contact: Dr. Sarah Johnson, PhD. Email: sarah.j@university.edu, Phone: (555) 123-4567. Department: Computer Science, Office: Room 401",
        "prompt": "Extract contact information as JSON: name, title, email, phone, department, office",
        "expected_keys": ["name", "title", "email", "phone", "department", "office"]
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


def extract_json(text: str) -> tuple:
    """Extract JSON from text, handling code blocks"""
    # Try to find JSON in code blocks first
    code_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if code_match:
        text = code_match.group(1)

    # Try to find JSON object or array
    for pattern in [r'\{[\s\S]*\}', r'\[[\s\S]*\]']:
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group()), True
            except:
                continue
    return None, False


def validate_json_structure(data: Any, task: Dict) -> Dict[str, Any]:
    """Validate JSON against task requirements"""
    results = {"valid": True, "errors": []}

    if task.get("is_array"):
        if not isinstance(data, list):
            results["valid"] = False
            results["errors"].append("Expected array")
            return results
        if len(data) < task.get("min_items", 1):
            results["valid"] = False
            results["errors"].append(f"Expected at least {task['min_items']} items")
        if task.get("item_keys") and data:
            for i, item in enumerate(data[:3]):
                if isinstance(item, dict):
                    missing = [k for k in task["item_keys"] if k not in item]
                    if missing:
                        results["errors"].append(f"Item {i} missing keys: {missing}")
    else:
        if not isinstance(data, dict):
            results["valid"] = False
            results["errors"].append("Expected object")
            return results

        if task.get("required_keys"):
            missing = [k for k in task["required_keys"] if k not in data]
            if missing:
                results["valid"] = False
                results["errors"].append(f"Missing keys: {missing}")

        if task.get("nested_checks"):
            for parent, children in task["nested_checks"].items():
                if parent in data and isinstance(data[parent], dict):
                    missing = [k for k in children if k not in data[parent]]
                    if missing:
                        results["errors"].append(f"{parent} missing: {missing}")

    return results


def run_json_generation_benchmark() -> Dict[str, Any]:
    print("\n  Running JSON Generation benchmark...")
    results = []
    system = "You are a helpful assistant. Always respond with valid JSON only, no explanations."

    for task in JSON_TASKS:
        print(f"    [{task['id']}] {task['name']}...")
        response = call_llm(task["prompt"], system)

        if response["success"]:
            data, parsed = extract_json(response["content"])
            validation = validate_json_structure(data, task) if parsed else {"valid": False, "errors": ["Failed to parse"]}

            result = {
                "task_id": task["id"],
                "name": task["name"],
                "success": True,
                "json_valid": parsed,
                "structure_valid": validation["valid"],
                "errors": validation["errors"],
                "latency": response["latency"]
            }
        else:
            result = {"task_id": task["id"], "success": False, "json_valid": False, "structure_valid": False}

        results.append(result)
        status = "VALID" if result.get("structure_valid") else ("PARSED" if result.get("json_valid") else "FAIL")
        print(f"      {status}")

    valid_json = sum(1 for r in results if r.get("json_valid"))
    valid_structure = sum(1 for r in results if r.get("structure_valid"))

    return {
        "summary": {
            "benchmark": "JSON Generation",
            "total": len(JSON_TASKS),
            "valid_json_rate": valid_json / len(JSON_TASKS),
            "valid_structure_rate": valid_structure / len(JSON_TASKS)
        },
        "results": results
    }


def run_schema_following_benchmark() -> Dict[str, Any]:
    print("\n  Running Schema Following benchmark...")
    results = []
    system = "You are a helpful assistant. Generate valid JSON that strictly follows the given schema. Output only JSON."

    for task in SCHEMA_TASKS:
        print(f"    [{task['id']}] {task['name']}...")
        response = call_llm(task["prompt"], system)

        if response["success"]:
            data, parsed = extract_json(response["content"])
            schema_valid = False

            if parsed and isinstance(data, dict):
                required = task["schema"].get("required", [])
                schema_valid = all(k in data for k in required)

            result = {
                "task_id": task["id"],
                "name": task["name"],
                "success": True,
                "json_valid": parsed,
                "schema_valid": schema_valid,
                "latency": response["latency"]
            }
        else:
            result = {"task_id": task["id"], "success": False, "json_valid": False, "schema_valid": False}

        results.append(result)
        status = "VALID" if result.get("schema_valid") else "FAIL"
        print(f"      {status}")

    schema_valid = sum(1 for r in results if r.get("schema_valid"))

    return {
        "summary": {
            "benchmark": "Schema Following",
            "total": len(SCHEMA_TASKS),
            "schema_compliance_rate": schema_valid / len(SCHEMA_TASKS)
        },
        "results": results
    }


def run_extraction_benchmark() -> Dict[str, Any]:
    print("\n  Running Data Extraction benchmark...")
    results = []
    system = "You are a data extraction assistant. Extract information from text and return as valid JSON only."

    for task in EXTRACTION_TASKS:
        print(f"    [{task['id']}] {task['name']}...")
        prompt = f"Text: {task['input']}\n\nTask: {task['prompt']}"
        response = call_llm(prompt, system)

        if response["success"]:
            data, parsed = extract_json(response["content"])
            keys_found = 0
            if parsed and isinstance(data, dict):
                keys_found = sum(1 for k in task["expected_keys"] if k in data)

            result = {
                "task_id": task["id"],
                "name": task["name"],
                "success": True,
                "json_valid": parsed,
                "keys_found": keys_found,
                "keys_expected": len(task["expected_keys"]),
                "extraction_rate": keys_found / len(task["expected_keys"]) if parsed else 0,
                "latency": response["latency"]
            }
        else:
            result = {"task_id": task["id"], "success": False, "json_valid": False, "extraction_rate": 0}

        results.append(result)
        print(f"      Keys: {result.get('keys_found', 0)}/{result.get('keys_expected', 0)}")

    avg_extraction = sum(r.get("extraction_rate", 0) for r in results) / len(results)

    return {
        "summary": {
            "benchmark": "Data Extraction",
            "total": len(EXTRACTION_TASKS),
            "avg_extraction_rate": avg_extraction
        },
        "results": results
    }


def main():
    print("=" * 70)
    print("  ARBM TRACK 11: STRUCTURED OUTPUT & JSON GENERATION")
    print("  Model: Llama-3-8B-Instruct via vLLM")
    print("=" * 70)

    json_results = run_json_generation_benchmark()
    schema_results = run_schema_following_benchmark()
    extract_results = run_extraction_benchmark()

    print("\n" + "=" * 70)
    print("  TRACK 11 SUMMARY")
    print("=" * 70)
    print(f"\n  JSON Generation:     {json_results['summary']['valid_structure_rate']*100:.1f}% valid")
    print(f"  Schema Following:    {schema_results['summary']['schema_compliance_rate']*100:.1f}% compliant")
    print(f"  Data Extraction:     {extract_results['summary']['avg_extraction_rate']*100:.1f}% keys extracted")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "track": "11_json_output",
        "model": "llama-3-8b-instruct",
        "benchmarks": {
            "json_generation": json_results,
            "schema_following": schema_results,
            "data_extraction": extract_results
        }
    }

    outfile = f"{OUTPUT_DIR}/track_11_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved: {outfile}")
    print("=" * 70)


if __name__ == "__main__":
    main()
