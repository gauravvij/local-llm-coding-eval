#!/usr/bin/env python3
"""
Function Calling / Tool Use Evaluation Script
Evaluates models on their ability to correctly call functions/tools.
"""

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable
import requests


@dataclass
class ToolCallResult:
    """Result of a single tool call evaluation."""
    task_id: str
    model: str
    prompt: str
    expected_tool: str
    expected_params: Dict[str, Any]
    predicted_tool: Optional[str]
    predicted_params: Optional[Dict[str, Any]]
    correct_tool: bool
    correct_params: bool
    response_time: float
    raw_response: str


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
    
    def generate(self, model: str, prompt: str, temperature: float = 0.1,
                 max_tokens: int = 2048, system: Optional[str] = None) -> str:
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
        # don't exhaust the budget before the actual JSON tool call is generated
        num_predict = 4096 if "qwen3.6" in model else 2048
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


class ToolCallingEvaluator:
    """Evaluates function calling / tool use capabilities."""
    
    def __init__(self, models: List[str], client: OllamaClient):
        self.models = models
        self.client = client
        self.results: Dict[str, List[ToolCallResult]] = {model: [] for model in models}
    
    def parse_tool_call(self, response: str) -> tuple:
        """Parse tool call from model response."""
        # Try to extract JSON tool call from markdown code blocks
        json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        json_matches = re.findall(json_pattern, response, re.DOTALL)
        
        for match in json_matches:
            try:
                data = json.loads(match.strip())
                if "name" in data or "tool" in data or "function" in data:
                    tool_name = data.get("name") or data.get("tool") or data.get("function")
                    params = data.get("parameters") or data.get("params") or data.get("arguments") or {}
                    if not params and "args" in data:
                        params = data["args"]
                    return tool_name, params
            except json.JSONDecodeError:
                continue
        
        # Try to parse raw JSON directly (models often output JSON without markdown fences)
        try:
            # Look for JSON objects in the response
            json_obj_pattern = r'\{[^{}]*"name"[^{}]*\}'
            json_matches = re.findall(json_obj_pattern, response, re.DOTALL)
            for match in json_matches:
                try:
                    data = json.loads(match.strip())
                    if "name" in data or "tool" in data or "function" in data:
                        tool_name = data.get("name") or data.get("tool") or data.get("function")
                        params = data.get("parameters") or data.get("params") or data.get("arguments") or {}
                        if not params and "args" in data:
                            params = data["args"]
                        return tool_name, params
                except json.JSONDecodeError:
                    continue
            
            # Also try parsing the entire response as JSON
            data = json.loads(response.strip())
            if "name" in data or "tool" in data or "function" in data:
                tool_name = data.get("name") or data.get("tool") or data.get("function")
                params = data.get("parameters") or data.get("params") or data.get("arguments") or {}
                if not params and "args" in data:
                    params = data["args"]
                return tool_name, params
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Try to find function call pattern: function_name(arg1=value1, arg2=value2)
        func_pattern = r'(\w+)\s*\((.*?)\)'
        func_matches = re.findall(func_pattern, response, re.DOTALL)
        
        for match in func_matches:
            tool_name = match[0]
            args_str = match[1]
            params = {}
            
            # Parse key=value pairs
            kv_pattern = r'(\w+)\s*=\s*([^,]+)'
            kv_matches = re.findall(kv_pattern, args_str)
            for key, value in kv_matches:
                value = value.strip().strip('"\'')
                # Try to convert to number if possible
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass
                params[key] = value
            
            return tool_name, params
        
        # Try to find XML-style tool call
        xml_pattern = r'<tool>\s*<name>(.*?)</name>\s*<parameters>(.*?)</parameters>\s*</tool>'
        xml_matches = re.findall(xml_pattern, response, re.DOTALL)
        
        for match in xml_matches:
            tool_name = match[0].strip()
            try:
                params = json.loads(match[1].strip())
            except json.JSONDecodeError:
                params = {}
            return tool_name, params
        
        return None, None
    
    def params_match(self, expected: Dict[str, Any], predicted: Dict[str, Any]) -> bool:
        """Check if parameters match."""
        if not expected and not predicted:
            return True
        if not expected or not predicted:
            return False
        
        # Check all expected keys are present with correct values
        for key, value in expected.items():
            if key not in predicted:
                return False
            # Allow type coercion for numbers
            if isinstance(value, (int, float)) and isinstance(predicted[key], (int, float)):
                if abs(value - predicted[key]) > 0.001:
                    return False
            elif str(predicted[key]).lower() != str(value).lower():
                return False
        
        return True
    
    def evaluate_task(self, model: str, task: Dict[str, Any]) -> ToolCallResult:
        """Evaluate a single tool calling task."""
        task_id = task.get("task_id", "unknown")
        prompt = task.get("prompt", "")
        system_prompt = task.get("system_prompt", "")
        expected_tool = task.get("expected_tool", "")
        expected_params = task.get("expected_params", {})
        
        # Generate response
        start_time = time.time()
        response = self.client.generate(model, prompt, system=system_prompt)
        response_time = time.time() - start_time
        
        # Parse tool call
        predicted_tool, predicted_params = self.parse_tool_call(response)
        
        # Check correctness
        correct_tool = predicted_tool == expected_tool
        correct_params = self.params_match(expected_params, predicted_params or {})
        
        return ToolCallResult(
            task_id=task_id,
            model=model,
            prompt=prompt,
            expected_tool=expected_tool,
            expected_params=expected_params,
            predicted_tool=predicted_tool,
            predicted_params=predicted_params,
            correct_tool=correct_tool,
            correct_params=correct_params,
            response_time=response_time,
            raw_response=response
        )
    
    def run_evaluation(self, tasks: List[Dict[str, Any]]) -> Dict[str, List[ToolCallResult]]:
        """Run evaluation on all models and tasks."""
        for model in self.models:
            print(f"\n{'='*60}")
            print(f"Evaluating model: {model}")
            print(f"{'='*60}")
            
            correct_tool_count = 0
            correct_params_count = 0
            total_count = len(tasks)
            
            for i, task in enumerate(tasks):
                print(f"  Task {i+1}/{total_count}: {task.get('task_id', 'unknown')}...", end=" ")
                
                result = self.evaluate_task(model, task)
                self.results[model].append(result)
                
                if result.correct_tool:
                    correct_tool_count += 1
                    tool_status = "✓"
                else:
                    tool_status = "✗"
                
                if result.correct_params:
                    correct_params_count += 1
                    params_status = "✓"
                else:
                    params_status = "✗"
                
                print(f"Tool: {tool_status} Params: {params_status}")
            
            tool_accuracy = correct_tool_count / total_count if total_count > 0 else 0
            params_accuracy = correct_params_count / total_count if total_count > 0 else 0
            print(f"\n  Results for {model}:")
            print(f"    Tool selection: {correct_tool_count}/{total_count} ({tool_accuracy:.2%})")
            print(f"    Parameters:     {correct_params_count}/{total_count} ({params_accuracy:.2%})")
        
        return self.results
    
    def save_results(self, output_dir: str):
        """Save evaluation results to JSON."""
        os.makedirs(output_dir, exist_ok=True)
        
        for model, results in self.results.items():
            safe_model_name = model.replace(":", "_").replace("/", "_")
            output_path = os.path.join(output_dir, f"tool_calling_{safe_model_name}.json")
            
            serializable_results = []
            for r in results:
                serializable_results.append({
                    "task_id": r.task_id,
                    "model": r.model,
                    "prompt": r.prompt,
                    "expected_tool": r.expected_tool,
                    "expected_params": r.expected_params,
                    "predicted_tool": r.predicted_tool,
                    "predicted_params": r.predicted_params,
                    "correct_tool": r.correct_tool,
                    "correct_params": r.correct_params,
                    "response_time": r.response_time,
                    "raw_response": r.raw_response
                })
            
            correct_tools = sum(1 for r in results if r.correct_tool)
            correct_params = sum(1 for r in results if r.correct_params)
            
            with open(output_path, 'w') as f:
                json.dump({
                    "model": model,
                    "total_tasks": len(results),
                    "correct_tools": correct_tools,
                    "correct_params": correct_params,
                    "tool_accuracy": correct_tools / len(results) if results else 0,
                    "params_accuracy": correct_params / len(results) if results else 0,
                    "results": serializable_results
                }, f, indent=2)
            
            print(f"Saved results to {output_path}")


def load_tool_calling_tasks() -> List[Dict[str, Any]]:
    """Load tool calling evaluation tasks."""
    
    system_prompt = """You are a helpful assistant that can use tools to help users.
When you need to use a tool, respond with a JSON object in this format:
{
    "name": "tool_name",
    "parameters": {
        "param1": "value1",
        "param2": "value2"
    }
}

Available tools:
- get_weather(location: str, unit: str = "celsius"): Get weather for a location
- search_web(query: str, num_results: int = 5): Search the web
- calculate(expression: str): Calculate a mathematical expression
- send_email(to: str, subject: str, body: str): Send an email
- set_reminder(task: str, time: str): Set a reminder
- get_stock_price(symbol: str): Get current stock price
- translate_text(text: str, target_language: str): Translate text
- create_calendar_event(title: str, date: str, time: str): Create a calendar event
"""
    
    tasks = [
        {
            "task_id": "tool_1",
            "system_prompt": system_prompt,
            "prompt": "What's the weather like in New York?",
            "expected_tool": "get_weather",
            "expected_params": {"location": "New York"}
        },
        {
            "task_id": "tool_2",
            "system_prompt": system_prompt,
            "prompt": "Search for information about Python programming",
            "expected_tool": "search_web",
            "expected_params": {"query": "Python programming"}
        },
        {
            "task_id": "tool_3",
            "system_prompt": system_prompt,
            "prompt": "Calculate 15 * 23 + 7",
            "expected_tool": "calculate",
            "expected_params": {"expression": "15 * 23 + 7"}
        },
        {
            "task_id": "tool_4",
            "system_prompt": system_prompt,
            "prompt": "Send an email to john@example.com with subject 'Meeting' and body 'See you at 3pm'",
            "expected_tool": "send_email",
            "expected_params": {"to": "john@example.com", "subject": "Meeting", "body": "See you at 3pm"}
        },
        {
            "task_id": "tool_5",
            "system_prompt": system_prompt,
            "prompt": "Set a reminder to buy groceries tomorrow at 5pm",
            "expected_tool": "set_reminder",
            "expected_params": {"task": "buy groceries", "time": "tomorrow at 5pm"}
        },
        {
            "task_id": "tool_6",
            "system_prompt": system_prompt,
            "prompt": "What's the current stock price of AAPL?",
            "expected_tool": "get_stock_price",
            "expected_params": {"symbol": "AAPL"}
        },
        {
            "task_id": "tool_7",
            "system_prompt": system_prompt,
            "prompt": "Translate 'Hello, how are you?' to Spanish",
            "expected_tool": "translate_text",
            "expected_params": {"text": "Hello, how are you?", "target_language": "Spanish"}
        },
        {
            "task_id": "tool_8",
            "system_prompt": system_prompt,
            "prompt": "Create a calendar event titled 'Team Meeting' on 2024-12-25 at 10:00",
            "expected_tool": "create_calendar_event",
            "expected_params": {"title": "Team Meeting", "date": "2024-12-25", "time": "10:00"}
        },
        {
            "task_id": "tool_9",
            "system_prompt": system_prompt,
            "prompt": "Get the weather in Tokyo in Fahrenheit",
            "expected_tool": "get_weather",
            "expected_params": {"location": "Tokyo", "unit": "fahrenheit"}
        },
        {
            "task_id": "tool_10",
            "system_prompt": system_prompt,
            "prompt": "Search for 'machine learning tutorials' and return 10 results",
            "expected_tool": "search_web",
            "expected_params": {"query": "machine learning tutorials", "num_results": 10}
        }
    ]
    return tasks


def load_complex_tool_tasks() -> List[Dict[str, Any]]:
    """Load complex multi-step tool calling tasks."""
    
    system_prompt = """You are a helpful assistant that can use tools to help users.
When you need to use a tool, respond with a JSON object in this format:
{
    "name": "tool_name",
    "parameters": {
        "param1": "value1",
        "param2": "value2"
    }
}

Available tools:
- get_weather(location: str, unit: str = "celsius"): Get weather for a location
- search_web(query: str, num_results: int = 5): Search the web
- calculate(expression: str): Calculate a mathematical expression
- send_email(to: str, subject: str, body: str): Send an email
- set_reminder(task: str, time: str): Set a reminder
- get_stock_price(symbol: str): Get current stock price
- translate_text(text: str, target_language: str): Translate text
- create_calendar_event(title: str, date: str, time: str): Create a calendar event
"""
    
    tasks = [
        {
            "task_id": "complex_1",
            "system_prompt": system_prompt,
            "prompt": "I need to schedule a meeting with the team. Create a calendar event for 'Sprint Planning' on January 15th at 2pm, and send an email to team@company.com with the meeting details.",
            "expected_tool": "create_calendar_event",
            "expected_params": {"title": "Sprint Planning", "date": "January 15th", "time": "2pm"}
        },
        {
            "task_id": "complex_2",
            "system_prompt": system_prompt,
            "prompt": "I'm planning a trip to Paris. What's the weather there? Also, search for best restaurants in Paris.",
            "expected_tool": "get_weather",
            "expected_params": {"location": "Paris"}
        },
        {
            "task_id": "complex_3",
            "system_prompt": system_prompt,
            "prompt": "Calculate the total cost: 5 items at $12.99 each plus 8% tax",
            "expected_tool": "calculate",
            "expected_params": {"expression": "5 * 12.99 * 1.08"}
        }
    ]
    return tasks


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Function Calling / Tool Use Evaluation")
    parser.add_argument("--models", nargs="+", default=["qwen3.6:27b", "qwen3-coder:30b", "deepseek-coder:33b", "qwen3-coder:latest"],
                        help="Models to evaluate")
    parser.add_argument("--output-dir", default="/root/local_coding_eval/results",
                        help="Output directory for results")
    parser.add_argument("--task-type", choices=["simple", "complex", "both"], default="both",
                        help="Type of tasks to evaluate")
    args = parser.parse_args()
    
    # Initialize client and evaluator
    client = OllamaClient()
    evaluator = ToolCallingEvaluator(args.models, client)
    
    # Load tasks
    all_tasks = []
    if args.task_type in ["simple", "both"]:
        all_tasks.extend(load_tool_calling_tasks())
    if args.task_type in ["complex", "both"]:
        all_tasks.extend(load_complex_tool_tasks())
    
    print(f"Loaded {len(all_tasks)} tool calling tasks for evaluation")
    print(f"Models: {args.models}")
    
    # Run evaluation
    results = evaluator.run_evaluation(all_tasks)
    
    # Save results
    evaluator.save_results(args.output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("TOOL CALLING EVALUATION SUMMARY")
    print("="*60)
    for model in args.models:
        model_results = results[model]
        correct_tools = sum(1 for r in model_results if r.correct_tool)
        correct_params = sum(1 for r in model_results if r.correct_params)
        total = len(model_results)
        tool_acc = correct_tools / total if total > 0 else 0
        params_acc = correct_params / total if total > 0 else 0
        print(f"{model:30s}: Tool={tool_acc:.2%}, Params={params_acc:.2%}")


if __name__ == "__main__":
    main()
