#!/usr/bin/env python3
"""
ARBM Benchmark using Transformers (No vLLM dependency)
"""
import torch
import os
import json
import time
from datetime import datetime

def main():
    print("=" * 60)
    print("  ARBM Benchmark - Transformers")
    print("=" * 60)
    
    print("\nLoading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_path = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print("Model loaded successfully!")
    print("Device map:", model.hf_device_map)
    
    # Test prompts
    prompts = [
        "What is 2+2? Answer with just the number.",
        "Explain quantum computing in one sentence.",
        "Write a Python function to calculate factorial.",
        "What is the capital of France?",
        "Solve: If x + 5 = 12, what is x?",
    ]
    
    results = []
    total_time = 0
    
    print("\n" + "=" * 60)
    print("  Running Benchmarks")
    print("=" * 60)
    
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Prompt: {prompt[:50]}...")
        
        start = time.time()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        elapsed = time.time() - start
        total_time += elapsed
        
        print(f"Response: {response[:150]}...")
        print(f"Time: {elapsed:.2f}s")
        
        results.append({
            "prompt": prompt,
            "response": response,
            "time_seconds": elapsed,
            "tokens_generated": len(outputs[0]) - len(inputs.input_ids[0])
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"Total prompts: {len(prompts)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per prompt: {total_time/len(prompts):.2f}s")
    
    # Save results
    output_dir = "/mnt/fss/ARBM/benchmarks/results"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"transformers_benchmark_{timestamp}.json")
    
    output = {
        "timestamp": timestamp,
        "model": model_path,
        "total_prompts": len(prompts),
        "total_time_seconds": total_time,
        "avg_time_per_prompt": total_time / len(prompts),
        "results": results
    }
    
    with open(result_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {result_file}")
    print("\n=== Benchmark Complete ===")

if __name__ == "__main__":
    main()
