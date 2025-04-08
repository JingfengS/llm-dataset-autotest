import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/workspace/deepseek-model/DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--data_path", default="/workspace/datasets/math-500/test.jsonl")
    parser.add_argument("--output_dir", default="/workspace/test/math_500_test/resultNew")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=2500)
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser.parse_args()

def load_math500_data(file_path):
    """Load MATH-500 dataset from JSONL file"""
    tasks = []
    with open(file_path, 'r') as f:
        for line in f:
            tasks.append(json.loads(line))
    return tasks

def generate_prompt(problem):
    """Construct prompt for the problem"""
    return f"""<|beginoftext|>system
You are an advanced AI capable of solving challenging mathematical problems step-by-step. Please provide a complete solution followed by the final answer.

<|file_separator|>user
Problem: {problem.strip()}

Provide ONLY the final answer after the solution steps.<|file_separator|>
assistant
"""

def extract_answer(response):
    """Extract the final answer from model response"""
    return response.split("assistant")[-1].strip()

def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Load and preprocess dataset
    tasks = load_math500_data(args.data_path)
    if args.max_samples:
        tasks = tasks[:args.max_samples]

    summary = {}

    for task in tqdm(tasks, desc="Processing MATH-500"):
        try:
            full_prompt = generate_prompt(task['problem'])

            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

            generated_answer = extract_answer(response)

            # Compare generated answer with the dataset's answer
            validation_result = {
                "problem": task['problem'],
                "expected_answer": task['answer'],
                "generated_answer": generated_answer,
                "subject": task['subject'],
                "level": task['level'],
                "unique_id": task['unique_id'],
                "passed": generated_answer.strip() == task['answer'].strip()
            }

            # Save individual result
            output_path = Path(args.output_dir) / f"{task['unique_id'].replace('/', '_')}.json"
            with open(output_path, 'w') as f:
                json.dump(validation_result, f, indent=2)

            summary[task['unique_id']] = validation_result

        except Exception as e:
            print(f"Error processing task {task['unique_id']}: {str(e)}")
            summary[task['unique_id']] = {"error": str(e)}

    # Generate summary report
    with open(Path(args.output_dir) / "summary.json", 'w') as f:
        json.dump({
            "total_tasks": len(tasks),
            "passed_tasks": sum([v.get('passed', False) for v in summary.values()]),
            "detailed_summary": summary
        }, f, indent=2)

if __name__ == "__main__":
    main()
