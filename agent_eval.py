#!/usr/bin/env python3
"""
Agent Capabilities Evaluation Script
Evaluates models on multi-step reasoning, planning, and agent-like tasks.
"""

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import requests


@dataclass
class AgentTaskResult:
    """Result of a single agent task evaluation."""
    task_id: str
    model: str
    prompt: str
    expected_steps: List[str]
    predicted_steps: List[str]
    reasoning_quality: int  # 0-3 scale
    plan_correctness: float  # 0.0-1.0
    final_answer_correct: bool
    response_time: float
    raw_response: str


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
    
    def generate(self, model: str, prompt: str, temperature: float = 0.3,
                 max_tokens: int = 4096, system: Optional[str] = None) -> str:
        """Generate response from the model."""
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
        if system:
            payload["system"] = system
        
        # Use larger token budget for thinking models (qwen3.6) so <think> blocks
        # don't exhaust the budget before the actual answer is generated
        num_predict = 8192 if "qwen3.6" in model else 4096
        payload["options"]["num_predict"] = num_predict
        timeout_seconds = 1200 if "qwen3.6" in model else 600
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


class AgentEvaluator:
    """Evaluates agent capabilities."""
    
    def __init__(self, models: List[str], client: OllamaClient):
        self.models = models
        self.client = client
        self.results: Dict[str, List[AgentTaskResult]] = {model: [] for model in models}
    
    def extract_steps(self, response: str) -> List[str]:
        """Extract reasoning steps from response."""
        steps = []
        
        # Look for numbered steps
        numbered_pattern = r'(?:^|\n)\s*(?:Step\s*\d+[.:]|\d+[.)])\s*(.+?)(?=\n\s*(?:Step\s*\d+[.:]|\d+[.)])|\Z)'
        matches = re.findall(numbered_pattern, response, re.DOTALL | re.IGNORECASE)
        if matches:
            steps = [m.strip() for m in matches if m.strip()]
        
        # If no numbered steps, look for bullet points
        if not steps:
            bullet_pattern = r'(?:^|\n)\s*[-•*]\s*(.+?)(?=\n\s*[-•*]|\Z)'
            matches = re.findall(bullet_pattern, response, re.DOTALL)
            steps = [m.strip() for m in matches if m.strip()]
        
        return steps
    
    def extract_final_answer(self, response: str) -> str:
        """Extract final answer from response."""
        # Look for explicit answer markers
        patterns = [
            r'(?:Final Answer|Answer|Result)[:\s]+(.+?)(?=\n\n|\Z)',
            r'(?:Therefore|Thus|So)[:\s]+(.+?)(?=\n\n|\Z)',
            r'```\s*\n?(.+?)\n?```',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Return last paragraph if no explicit marker
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        if paragraphs:
            return paragraphs[-1]
        
        return response.strip()
    
    def evaluate_reasoning_quality(self, response: str, expected_steps: List[str]) -> int:
        """Evaluate reasoning quality on a 0-3 scale."""
        score = 0
        
        # Check for step-by-step reasoning
        steps = self.extract_steps(response)
        if len(steps) >= 2:
            score += 1
        
        # Check for logical flow
        logical_markers = ['first', 'then', 'next', 'after', 'finally', 'because', 'therefore', 'thus']
        has_logical_flow = any(marker in response.lower() for marker in logical_markers)
        if has_logical_flow:
            score += 1
        
        # Check for explanation of reasoning
        explanation_markers = ['need to', 'should', 'must', 'will', 'calculate', 'compute', 'find']
        has_explanation = any(marker in response.lower() for marker in explanation_markers)
        if has_explanation:
            score += 1
        
        return score
    
    def evaluate_plan_correctness(self, predicted_steps: List[str], expected_steps: List[str]) -> float:
        """Evaluate how correct the plan is (0.0-1.0)."""
        if not expected_steps:
            return 1.0 if not predicted_steps else 0.0
        
        if not predicted_steps:
            return 0.0
        
        # Check how many expected steps are covered
        covered = 0
        for expected in expected_steps:
            expected_lower = expected.lower()
            for predicted in predicted_steps:
                predicted_lower = predicted.lower()
                # Check for semantic similarity (simplified)
                if any(word in predicted_lower for word in expected_lower.split() if len(word) > 3):
                    covered += 1
                    break
        
        return covered / len(expected_steps)
    
    def check_answer_correctness(self, response: str, expected_answer: str) -> bool:
        """Check if the final answer is correct."""
        final_answer = self.extract_final_answer(response).lower()
        expected = expected_answer.lower()
        
        # Direct match
        if expected in final_answer or final_answer in expected:
            return True
        
        # Numeric match (allow small differences)
        try:
            expected_num = float(expected.replace(',', ''))
            # Extract numbers from response
            numbers = re.findall(r'-?\d+\.?\d*', final_answer.replace(',', ''))
            for num_str in numbers:
                if abs(float(num_str) - expected_num) < 0.01:
                    return True
        except ValueError:
            pass
        
        return False
    
    def evaluate_task(self, model: str, task: Dict[str, Any]) -> AgentTaskResult:
        """Evaluate a single agent task."""
        task_id = task.get("task_id", "unknown")
        prompt = task.get("prompt", "")
        system_prompt = task.get("system_prompt", "")
        expected_steps = task.get("expected_steps", [])
        expected_answer = task.get("expected_answer", "")
        
        # Generate response
        start_time = time.time()
        response = self.client.generate(model, prompt, system=system_prompt)
        response_time = time.time() - start_time
        
        # Extract and evaluate
        predicted_steps = self.extract_steps(response)
        reasoning_quality = self.evaluate_reasoning_quality(response, expected_steps)
        plan_correctness = self.evaluate_plan_correctness(predicted_steps, expected_steps)
        final_answer_correct = self.check_answer_correctness(response, expected_answer)
        
        return AgentTaskResult(
            task_id=task_id,
            model=model,
            prompt=prompt,
            expected_steps=expected_steps,
            predicted_steps=predicted_steps,
            reasoning_quality=reasoning_quality,
            plan_correctness=plan_correctness,
            final_answer_correct=final_answer_correct,
            response_time=response_time,
            raw_response=response
        )
    
    def run_evaluation(self, tasks: List[Dict[str, Any]]) -> Dict[str, List[AgentTaskResult]]:
        """Run evaluation on all models and tasks."""
        for model in self.models:
            print(f"\n{'='*60}")
            print(f"Evaluating model: {model}")
            print(f"{'='*60}")
            
            correct_answers = 0
            total_reasoning_score = 0
            total_plan_score = 0
            total_count = len(tasks)
            
            for i, task in enumerate(tasks):
                print(f"  Task {i+1}/{total_count}: {task.get('task_id', 'unknown')}...", end=" ")
                
                result = self.evaluate_task(model, task)
                self.results[model].append(result)
                
                if result.final_answer_correct:
                    correct_answers += 1
                    answer_status = "✓"
                else:
                    answer_status = "✗"
                
                total_reasoning_score += result.reasoning_quality
                total_plan_score += result.plan_correctness
                
                print(f"Answer: {answer_status} Reasoning: {result.reasoning_quality}/3 Plan: {result.plan_correctness:.2f}")
            
            answer_accuracy = correct_answers / total_count if total_count > 0 else 0
            avg_reasoning = total_reasoning_score / total_count if total_count > 0 else 0
            avg_plan = total_plan_score / total_count if total_count > 0 else 0
            
            print(f"\n  Results for {model}:")
            print(f"    Correct Answers: {correct_answers}/{total_count} ({answer_accuracy:.2%})")
            print(f"    Avg Reasoning:   {avg_reasoning:.2f}/3.0")
            print(f"    Avg Plan Score:  {avg_plan:.2%}")
        
        return self.results
    
    def save_results(self, output_dir: str):
        """Save evaluation results to JSON."""
        os.makedirs(output_dir, exist_ok=True)
        
        for model, results in self.results.items():
            safe_model_name = model.replace(":", "_").replace("/", "_")
            output_path = os.path.join(output_dir, f"agent_{safe_model_name}.json")
            
            serializable_results = []
            for r in results:
                serializable_results.append({
                    "task_id": r.task_id,
                    "model": r.model,
                    "prompt": r.prompt,
                    "expected_steps": r.expected_steps,
                    "predicted_steps": r.predicted_steps,
                    "reasoning_quality": r.reasoning_quality,
                    "plan_correctness": r.plan_correctness,
                    "final_answer_correct": r.final_answer_correct,
                    "response_time": r.response_time,
                    "raw_response": r.raw_response
                })
            
            correct = sum(1 for r in results if r.final_answer_correct)
            avg_reasoning = sum(r.reasoning_quality for r in results) / len(results) if results else 0
            avg_plan = sum(r.plan_correctness for r in results) / len(results) if results else 0
            
            with open(output_path, 'w') as f:
                json.dump({
                    "model": model,
                    "total_tasks": len(results),
                    "correct_answers": correct,
                    "answer_accuracy": correct / len(results) if results else 0,
                    "avg_reasoning_score": avg_reasoning,
                    "avg_plan_score": avg_plan,
                    "results": serializable_results
                }, f, indent=2)
            
            print(f"Saved results to {output_path}")


def load_agent_tasks() -> List[Dict[str, Any]]:
    """Load agent evaluation tasks."""
    
    system_prompt = """You are an intelligent agent that can solve complex problems through step-by-step reasoning.
Break down problems into clear steps and explain your reasoning.
Provide a final answer at the end."""
    
    tasks = [
        {
            "task_id": "agent_1",
            "system_prompt": system_prompt,
            "prompt": "A store is having a sale where all items are 20% off. If a customer buys a $150 item and a $80 item, and there's an additional 8% sales tax, what is the final total?",
            "expected_steps": [
                "calculate discount on first item",
                "calculate discount on second item",
                "sum discounted prices",
                "calculate tax",
                "add tax to total"
            ],
            "expected_answer": "198.72"
        },
        {
            "task_id": "agent_2",
            "system_prompt": system_prompt,
            "prompt": "A train travels 120 km in 2 hours, then stops for 30 minutes, then continues for another 180 km in 3 hours. What is the average speed for the entire journey (excluding stop time)?",
            "expected_steps": [
                "calculate total distance",
                "calculate total moving time",
                "divide distance by time"
            ],
            "expected_answer": "60"
        },
        {
            "task_id": "agent_3",
            "system_prompt": system_prompt,
            "prompt": "You have $500 to invest. Option A gives 5% annual return compounded monthly. Option B gives 4.8% annual return compounded quarterly. Which option gives more money after 2 years, and by how much?",
            "expected_steps": [
                "calculate compound interest for option A",
                "calculate compound interest for option B",
                "compare results",
                "calculate difference"
            ],
            "expected_answer": "A"
        },
        {
            "task_id": "agent_4",
            "system_prompt": system_prompt,
            "prompt": "A rectangular garden is 15 meters long and 8 meters wide. You want to build a path around it that is 1 meter wide. What is the area of the path only?",
            "expected_steps": [
                "calculate garden area",
                "calculate outer dimensions",
                "calculate total area with path",
                "subtract garden area from total"
            ],
            "expected_answer": "50"
        },
        {
            "task_id": "agent_5",
            "system_prompt": system_prompt,
            "prompt": "If 8 workers can build a wall in 10 days, how many days will it take 5 workers to build the same wall, assuming they work at the same rate?",
            "expected_steps": [
                "calculate total work in worker-days",
                "divide by new number of workers"
            ],
            "expected_answer": "16"
        },
        {
            "task_id": "agent_6",
            "system_prompt": system_prompt,
            "prompt": "A mixture contains alcohol and water in the ratio 3:2. If 5 liters of alcohol is added, the ratio becomes 2:1. What was the original volume of the mixture?",
            "expected_steps": [
                "set up equation for original mixture",
                "set up equation after adding alcohol",
                "solve for original volume"
            ],
            "expected_answer": "25"
        },
        {
            "task_id": "agent_7",
            "system_prompt": system_prompt,
            "prompt": "You need to schedule 3 tasks: Task A takes 2 hours, Task B takes 3 hours, Task C takes 1 hour. Task B depends on Task A. Task C can run anytime. What is the minimum time to complete all tasks if you can only do one at a time?",
            "expected_steps": [
                "identify dependencies",
                "determine order",
                "sum durations"
            ],
            "expected_answer": "6"
        },
        {
            "task_id": "agent_8",
            "system_prompt": system_prompt,
            "prompt": "A car travels from City A to City B at 60 km/h and returns at 40 km/h. The distance between cities is 120 km. What is the average speed for the entire round trip?",
            "expected_steps": [
                "calculate time to go",
                "calculate time to return",
                "calculate total distance",
                "divide total distance by total time"
            ],
            "expected_answer": "48"
        }
    ]
    return tasks


def load_complex_agent_tasks() -> List[Dict[str, Any]]:
    """Load complex multi-step agent tasks."""
    
    system_prompt = """You are an intelligent agent that can solve complex problems through step-by-step reasoning.
Break down problems into clear steps and explain your reasoning.
Provide a final answer at the end."""
    
    tasks = [
        {
            "task_id": "complex_agent_1",
            "system_prompt": system_prompt,
            "prompt": """You are planning a project with the following constraints:
- Task 1: 3 days (must be done first)
- Task 2: 2 days (depends on Task 1)
- Task 3: 4 days (depends on Task 1)
- Task 4: 2 days (depends on Task 2 and Task 3)
- Task 5: 1 day (depends on Task 4)

You have 2 workers who can work on different tasks simultaneously if dependencies allow.
What is the minimum time to complete the project?""",
            "expected_steps": [
                "identify critical path",
                "schedule parallel tasks",
                "calculate total duration"
            ],
            "expected_answer": "10"
        },
        {
            "task_id": "complex_agent_2",
            "system_prompt": system_prompt,
            "prompt": """A company produces two products X and Y.
- Product X requires 2 hours of machine time and 3 hours of labor, profit $50
- Product Y requires 1 hour of machine time and 4 hours of labor, profit $40
- Available: 100 machine hours, 180 labor hours

What is the maximum profit achievable?""",
            "expected_steps": [
                "set up constraints",
                "set up objective function",
                "find optimal solution"
            ],
            "expected_answer": "2600"
        }
    ]
    return tasks


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Agent Capabilities Evaluation")
    parser.add_argument("--models", nargs="+", default=["qwen3.6:27b", "qwen3-coder:30b", "deepseek-coder:33b", "qwen3-coder:latest"],
                        help="Models to evaluate")
    parser.add_argument("--output-dir", default="/root/local_coding_eval/results",
                        help="Output directory for results")
    parser.add_argument("--task-type", choices=["simple", "complex", "both"], default="both",
                        help="Type of tasks to evaluate")
    args = parser.parse_args()
    
    # Initialize client and evaluator
    client = OllamaClient()
    evaluator = AgentEvaluator(args.models, client)
    
    # Load tasks
    all_tasks = []
    if args.task_type in ["simple", "both"]:
        all_tasks.extend(load_agent_tasks())
    if args.task_type in ["complex", "both"]:
        all_tasks.extend(load_complex_agent_tasks())
    
    print(f"Loaded {len(all_tasks)} agent tasks for evaluation")
    print(f"Models: {args.models}")
    
    # Run evaluation
    results = evaluator.run_evaluation(all_tasks)
    
    # Save results
    evaluator.save_results(args.output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("AGENT EVALUATION SUMMARY")
    print("="*60)
    for model in args.models:
        model_results = results[model]
        correct = sum(1 for r in model_results if r.final_answer_correct)
        avg_reasoning = sum(r.reasoning_quality for r in model_results) / len(model_results) if model_results else 0
        avg_plan = sum(r.plan_correctness for r in model_results) / len(model_results) if model_results else 0
        total = len(model_results)
        acc = correct / total if total > 0 else 0
        print(f"{model:30s}: Acc={acc:.2%}, Reasoning={avg_reasoning:.2f}/3, Plan={avg_plan:.2%}")


if __name__ == "__main__":
    main()
