#!/usr/bin/env python3
"""
Code Generation Evaluation Script
Evaluates models on HumanEval-style and MBPP-style coding benchmarks.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import requests


@dataclass
class EvalResult:
    """Result of a single evaluation."""
    task_id: str
    model: str
    prompt: str
    generated_code: str
    expected_output: Any
    actual_output: Any
    passed: bool
    execution_time: float
    error: Optional[str] = None


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
    
    def generate(self, model: str, prompt: str, temperature: float = 0.2, 
                 max_tokens: int = 2048, stop: Optional[List[str]] = None) -> str:
        """Generate code from the model."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        if stop:
            payload["options"]["stop"] = stop
        
        # Use extended timeout for dense models like qwen3.6 to avoid timeout artifacts
        timeout_seconds = 1200 if "qwen3.6" in model else 600
        # Use larger token budget for thinking models (qwen3.6) to avoid exhausting
        # budget on <think> blocks before actual code is generated
        num_predict = 8192 if "qwen3.6" in model else 2048
        payload["options"]["num_predict"] = num_predict
        try:
            response = requests.post(url, json=payload, timeout=timeout_seconds)
            response.raise_for_status()
            raw = response.json().get("response", "")
            # Strip <think>...</think> blocks produced by reasoning models
            raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            return raw
        except Exception as e:
            print(f"Error generating from {model}: {e}")
            return ""


class CodeEvaluator:
    """Evaluates code generation capabilities."""
    
    def __init__(self, models: List[str], client: OllamaClient):
        self.models = models
        self.client = client
        self.results: Dict[str, List[EvalResult]] = {model: [] for model in models}
    
    def extract_code(self, text: str) -> str:
        """Extract code from model output."""
        # Try to extract code from markdown code blocks
        code_pattern = r"```(?:python)?\s*\n(.*?)\n```"
        matches = re.findall(code_pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        
        # Try to find function definition
        func_pattern = r"(def\s+\w+\s*\([^)]*\):.*?)(?=\n\n|\Z)"
        matches = re.findall(func_pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        return text.strip()
    
    def execute_code(self, code: str, test_code: str) -> tuple:
        """Execute code and return result."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Write the solution code
            f.write(code + "\n\n")
            # Write the test code
            f.write(test_code + "\n")
            temp_path = f.name
        
        try:
            start_time = time.time()
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                return True, result.stdout.strip(), execution_time
            else:
                return False, result.stderr.strip(), execution_time
        except subprocess.TimeoutExpired:
            return False, "Timeout", 10.0
        except Exception as e:
            return False, str(e), 0.0
        finally:
            os.unlink(temp_path)
    
    def evaluate_task(self, model: str, task: Dict[str, Any]) -> EvalResult:
        """Evaluate a single task."""
        task_id = task.get("task_id", "unknown")
        prompt = task.get("prompt", "")
        canonical_solution = task.get("canonical_solution", "")
        test_code = task.get("test", "")
        entry_point = task.get("entry_point", "")
        
        # Generate code
        start_time = time.time()
        generated = self.client.generate(model, prompt)
        generation_time = time.time() - start_time
        
        code = self.extract_code(generated)
        
        # Execute the code with tests
        passed, output, exec_time = self.execute_code(code, test_code)
        
        return EvalResult(
            task_id=task_id,
            model=model,
            prompt=prompt,
            generated_code=code,
            expected_output=canonical_solution,
            actual_output=output,
            passed=passed,
            execution_time=generation_time + exec_time,
            error=None if passed else output
        )
    
    def run_evaluation(self, tasks: List[Dict[str, Any]], max_workers: int = 4) -> Dict[str, List[EvalResult]]:
        """Run evaluation on all models and tasks."""
        for model in self.models:
            print(f"\n{'='*60}")
            print(f"Evaluating model: {model}")
            print(f"{'='*60}")
            
            passed_count = 0
            total_count = len(tasks)
            
            for i, task in enumerate(tasks):
                print(f"  Task {i+1}/{total_count}: {task.get('task_id', 'unknown')}...", end=" ")
                
                result = self.evaluate_task(model, task)
                self.results[model].append(result)
                
                if result.passed:
                    passed_count += 1
                    print("✓ PASS")
                else:
                    print(f"✗ FAIL: {result.error[:50] if result.error else 'Unknown error'}...")
            
            accuracy = passed_count / total_count if total_count > 0 else 0
            print(f"\n  Results for {model}: {passed_count}/{total_count} passed ({accuracy:.2%})")
        
        return self.results
    
    def save_results(self, output_dir: str):
        """Save evaluation results to JSON."""
        os.makedirs(output_dir, exist_ok=True)
        
        for model, results in self.results.items():
            safe_model_name = model.replace(":", "_").replace("/", "_")
            output_path = os.path.join(output_dir, f"code_gen_{safe_model_name}.json")
            
            serializable_results = []
            for r in results:
                serializable_results.append({
                    "task_id": r.task_id,
                    "model": r.model,
                    "prompt": r.prompt,
                    "generated_code": r.generated_code,
                    "expected_output": r.expected_output,
                    "actual_output": r.actual_output,
                    "passed": r.passed,
                    "execution_time": r.execution_time,
                    "error": r.error
                })
            
            with open(output_path, 'w') as f:
                json.dump({
                    "model": model,
                    "total_tasks": len(results),
                    "passed": sum(1 for r in results if r.passed),
                    "accuracy": sum(1 for r in results if r.passed) / len(results) if results else 0,
                    "results": serializable_results
                }, f, indent=2)
            
            print(f"Saved results to {output_path}")


def load_humaneval_tasks() -> List[Dict[str, Any]]:
    """Load HumanEval-style tasks."""
    # Simplified HumanEval tasks for demonstration
    tasks = [
        {
            "task_id": "humaneval_1",
            "prompt": "def has_close_elements(numbers: list[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
            "canonical_solution": "def has_close_elements(numbers: list[float], threshold: float) -> bool:\n    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n    return False\n",
            "test": "\ndef check(has_close_elements):\n    assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\n    assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n    assert has_close_elements([1.0, 2.0, 3.0], 0.0) == False\n    assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    print('All tests passed!')\n\ncheck(has_close_elements)\n",
            "entry_point": "has_close_elements"
        },
        {
            "task_id": "humaneval_2",
            "prompt": "from typing import List\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other.\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n",
            "canonical_solution": "from typing import List\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    paren_string = paren_string.replace(' ', '')\n    result = []\n    current = ''\n    depth = 0\n    for char in paren_string:\n        if char == '(':\n            depth += 1\n            current += char\n        elif char == ')':\n            depth -= 1\n            current += char\n            if depth == 0:\n                result.append(current)\n                current = ''\n    return result\n",
            "test": "\ndef check(separate_paren_groups):\n    assert separate_paren_groups('( ) (( )) (( )( ))') == ['()', '(())', '(()())']\n    assert separate_paren_groups('()') == ['()']\n    assert separate_paren_groups('(())(())') == ['(())', '(())']\n    assert separate_paren_groups('((()))') == ['((()))']\n    print('All tests passed!')\n\ncheck(separate_paren_groups)\n",
            "entry_point": "separate_paren_groups"
        },
        {
            "task_id": "humaneval_3",
            "prompt": "def truncate_number(number: float, decimals: int) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"\n",
            "canonical_solution": "def truncate_number(number: float, decimals: int = 0) -> float:\n    return round(number - int(number), decimals)\n",
            "test": "\ndef check(truncate_number):\n    assert abs(truncate_number(3.5) - 0.5) < 0.001\n    assert abs(truncate_number(2.7) - 0.7) < 0.001\n    assert abs(truncate_number(10.0) - 0.0) < 0.001\n    assert abs(truncate_number(0.12345) - 0.12345) < 0.001\n    print('All tests passed!')\n\ncheck(truncate_number)\n",
            "entry_point": "truncate_number"
        },
        {
            "task_id": "humaneval_4",
            "prompt": "from typing import List\n\ndef below_zero(operations: List[int]) -> bool:\n    \"\"\" You're given a list of deposit and withdrawal operations on a bank account that starts with\n    zero balance. Your task is to detect if at any point the balance of account falls below zero, and\n    at that point function should return True. Otherwise it should return False.\n    >>> below_zero([1, 2, 3])\n    False\n    >>> below_zero([1, 2, -4, 5])\n    True\n    \"\"\"\n",
            "canonical_solution": "from typing import List\n\ndef below_zero(operations: List[int]) -> bool:\n    balance = 0\n    for op in operations:\n        balance += op\n        if balance < 0:\n            return True\n    return False\n",
            "test": "\ndef check(below_zero):\n    assert below_zero([1, 2, 3]) == False\n    assert below_zero([1, 2, -4, 5]) == True\n    assert below_zero([1, 2, -3]) == False\n    assert below_zero([-1, 2, 3]) == True\n    assert below_zero([]) == False\n    print('All tests passed!')\n\ncheck(below_zero)\n",
            "entry_point": "below_zero"
        },
        {
            "task_id": "humaneval_5",
            "prompt": "def sum_squares(lst: list) -> int:\n    \"\"\" Return sum of squares of all even numbers and cubes of all odd numbers in the list.\n    >>> sum_squares([1, 2, 3, 4])\n    100\n    \"\"\"\n",
            "canonical_solution": "def sum_squares(lst: list) -> int:\n    total = 0\n    for num in lst:\n        if num % 2 == 0:\n            total += num ** 2\n        else:\n            total += num ** 3\n    return total\n",
            "test": "\ndef check(sum_squares):\n    assert sum_squares([1, 2, 3, 4]) == 1 + 4 + 27 + 16\n    assert sum_squares([2, 4, 6]) == 4 + 16 + 36\n    assert sum_squares([1, 3, 5]) == 1 + 27 + 125\n    assert sum_squares([]) == 0\n    print('All tests passed!')\n\ncheck(sum_squares)\n",
            "entry_point": "sum_squares"
        }
    ]
    return tasks


def load_mbpp_tasks() -> List[Dict[str, Any]]:
    """Load MBPP-style tasks."""
    tasks = [
        {
            "task_id": "mbpp_1",
            "prompt": "Write a function to find the sum of all numbers in a list.\n\ndef sum_list(numbers):\n    \"\"\"\n    Return the sum of all numbers in the list.\n    >>> sum_list([1, 2, 3, 4])\n    10\n    >>> sum_list([])\n    0\n    \"\"\"\n",
            "canonical_solution": "def sum_list(numbers):\n    return sum(numbers)\n",
            "test": "\ndef check(sum_list):\n    assert sum_list([1, 2, 3, 4]) == 10\n    assert sum_list([]) == 0\n    assert sum_list([5]) == 5\n    assert sum_list([-1, 1]) == 0\n    print('All tests passed!')\n\ncheck(sum_list)\n",
            "entry_point": "sum_list"
        },
        {
            "task_id": "mbpp_2",
            "prompt": "Write a function to check if a string is a palindrome.\n\ndef is_palindrome(s):\n    \"\"\"\n    Check if the string is a palindrome (reads same forwards and backwards).\n    >>> is_palindrome('racecar')\n    True\n    >>> is_palindrome('hello')\n    False\n    \"\"\"\n",
            "canonical_solution": "def is_palindrome(s):\n    return s == s[::-1]\n",
            "test": "\ndef check(is_palindrome):\n    assert is_palindrome('racecar') == True\n    assert is_palindrome('hello') == False\n    assert is_palindrome('a') == True\n    assert is_palindrome('') == True\n    assert is_palindrome('madam') == True\n    print('All tests passed!')\n\ncheck(is_palindrome)\n",
            "entry_point": "is_palindrome"
        },
        {
            "task_id": "mbpp_3",
            "prompt": "Write a function to find the factorial of a number.\n\ndef factorial(n):\n    \"\"\"\n    Return the factorial of n (n!).\n    >>> factorial(5)\n    120\n    >>> factorial(0)\n    1\n    \"\"\"\n",
            "canonical_solution": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n",
            "test": "\ndef check(factorial):\n    assert factorial(5) == 120\n    assert factorial(0) == 1\n    assert factorial(1) == 1\n    assert factorial(3) == 6\n    assert factorial(10) == 3628800\n    print('All tests passed!')\n\ncheck(factorial)\n",
            "entry_point": "factorial"
        },
        {
            "task_id": "mbpp_4",
            "prompt": "Write a function to find the maximum element in a list.\n\ndef find_max(lst):\n    \"\"\"\n    Return the maximum element in the list.\n    >>> find_max([1, 5, 3, 9, 2])\n    9\n    >>> find_max([-5, -2, -10])\n    -2\n    \"\"\"\n",
            "canonical_solution": "def find_max(lst):\n    return max(lst)\n",
            "test": "\ndef check(find_max):\n    assert find_max([1, 5, 3, 9, 2]) == 9\n    assert find_max([-5, -2, -10]) == -2\n    assert find_max([5]) == 5\n    assert find_max([1, 1, 1]) == 1\n    print('All tests passed!')\n\ncheck(find_max)\n",
            "entry_point": "find_max"
        },
        {
            "task_id": "mbpp_5",
            "prompt": "Write a function to count the frequency of each element in a list.\n\ndef count_frequency(lst):\n    \"\"\"\n    Return a dictionary with elements as keys and their frequencies as values.\n    >>> count_frequency([1, 2, 2, 3, 3, 3])\n    {1: 1, 2: 2, 3: 3}\n    >>> count_frequency([])\n    {}\n    \"\"\"\n",
            "canonical_solution": "def count_frequency(lst):\n    freq = {}\n    for item in lst:\n        freq[item] = freq.get(item, 0) + 1\n    return freq\n",
            "test": "\ndef check(count_frequency):\n    assert count_frequency([1, 2, 2, 3, 3, 3]) == {1: 1, 2: 2, 3: 3}\n    assert count_frequency([]) == {}\n    assert count_frequency(['a', 'b', 'a']) == {'a': 2, 'b': 1}\n    assert count_frequency([1, 1, 1, 1]) == {1: 4}\n    print('All tests passed!')\n\ncheck(count_frequency)\n",
            "entry_point": "count_frequency"
        }
    ]
    return tasks


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Code Generation Evaluation")
    parser.add_argument("--models", nargs="+", default=["qwen3.6:27b", "qwen3-coder:30b", "deepseek-coder:33b", "qwen3-coder:latest"],
                        help="Models to evaluate")
    parser.add_argument("--output-dir", default="/root/local_coding_eval/results",
                        help="Output directory for results")
    parser.add_argument("--benchmark", choices=["humaneval", "mbpp", "both"], default="both",
                        help="Which benchmark to run")
    args = parser.parse_args()
    
    # Initialize client and evaluator
    client = OllamaClient()
    evaluator = CodeEvaluator(args.models, client)
    
    # Load tasks
    all_tasks = []
    if args.benchmark in ["humaneval", "both"]:
        all_tasks.extend(load_humaneval_tasks())
    if args.benchmark in ["mbpp", "both"]:
        all_tasks.extend(load_mbpp_tasks())
    
    print(f"Loaded {len(all_tasks)} tasks for evaluation")
    print(f"Models: {args.models}")
    
    # Run evaluation
    results = evaluator.run_evaluation(all_tasks)
    
    # Save results
    evaluator.save_results(args.output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for model in args.models:
        model_results = results[model]
        passed = sum(1 for r in model_results if r.passed)
        total = len(model_results)
        accuracy = passed / total if total > 0 else 0
        print(f"{model:30s}: {passed}/{total} ({accuracy:.2%})")


if __name__ == "__main__":
    main()
