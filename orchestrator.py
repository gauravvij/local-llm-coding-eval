#!/usr/bin/env python3
"""
Orchestrator Script for Local LLM Evaluation Suite
Runs all evaluations and generates a comprehensive comparison report.
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any
import subprocess


class EvaluationOrchestrator:
    """Orchestrates all evaluations and generates reports."""
    
    def __init__(self, models: List[str], output_dir: str = "/root/local_coding_eval/results"):
        self.models = models
        self.output_dir = output_dir
        self.results_summary = {}
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def run_code_generation_eval(self) -> Dict[str, Any]:
        """Run code generation evaluation."""
        print("\n" + "="*70)
        print("RUNNING CODE GENERATION EVALUATION")
        print("="*70)
        
        script_path = "/root/local_coding_eval/code_generation_eval.py"
        cmd = [
            sys.executable, script_path,
            "--models"] + self.models + [
            "--output-dir", self.output_dir,
            "--benchmark", "both"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            return {"status": "success", "returncode": result.returncode}
        except subprocess.TimeoutExpired:
            print("Code generation evaluation timed out!")
            return {"status": "timeout", "returncode": -1}
        except Exception as e:
            print(f"Error running code generation eval: {e}")
            return {"status": "error", "returncode": -1, "error": str(e)}
    
    def run_function_calling_eval(self) -> Dict[str, Any]:
        """Run function calling evaluation."""
        print("\n" + "="*70)
        print("RUNNING FUNCTION CALLING EVALUATION")
        print("="*70)
        
        script_path = "/root/local_coding_eval/function_calling_eval.py"
        cmd = [
            sys.executable, script_path,
            "--models"] + self.models + [
            "--output-dir", self.output_dir,
            "--task-type", "both"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            return {"status": "success", "returncode": result.returncode}
        except subprocess.TimeoutExpired:
            print("Function calling evaluation timed out!")
            return {"status": "timeout", "returncode": -1}
        except Exception as e:
            print(f"Error running function calling eval: {e}")
            return {"status": "error", "returncode": -1, "error": str(e)}
    
    def run_agent_eval(self) -> Dict[str, Any]:
        """Run agent capabilities evaluation."""
        print("\n" + "="*70)
        print("RUNNING AGENT CAPABILITIES EVALUATION")
        print("="*70)
        
        script_path = "/root/local_coding_eval/agent_eval.py"
        cmd = [
            sys.executable, script_path,
            "--models"] + self.models + [
            "--output-dir", self.output_dir,
            "--task-type", "both"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            return {"status": "success", "returncode": result.returncode}
        except subprocess.TimeoutExpired:
            print("Agent evaluation timed out!")
            return {"status": "timeout", "returncode": -1}
        except Exception as e:
            print(f"Error running agent eval: {e}")
            return {"status": "error", "returncode": -1, "error": str(e)}
    
    def collect_results(self) -> Dict[str, Dict[str, Any]]:
        """Collect results from all evaluation files."""
        all_results = {}
        
        for model in self.models:
            safe_model_name = model.replace(":", "_").replace("/", "_")
            model_results = {
                "code_generation": {},
                "function_calling": {},
                "agent": {}
            }
            
            # Code generation results
            code_gen_path = os.path.join(self.output_dir, f"code_gen_{safe_model_name}.json")
            if os.path.exists(code_gen_path):
                try:
                    with open(code_gen_path, 'r') as f:
                        model_results["code_generation"] = json.load(f)
                except Exception as e:
                    print(f"Error loading code gen results for {model}: {e}")
            
            # Function calling results
            tool_path = os.path.join(self.output_dir, f"tool_calling_{safe_model_name}.json")
            if os.path.exists(tool_path):
                try:
                    with open(tool_path, 'r') as f:
                        model_results["function_calling"] = json.load(f)
                except Exception as e:
                    print(f"Error loading tool calling results for {model}: {e}")
            
            # Agent results
            agent_path = os.path.join(self.output_dir, f"agent_{safe_model_name}.json")
            if os.path.exists(agent_path):
                try:
                    with open(agent_path, 'r') as f:
                        model_results["agent"] = json.load(f)
                except Exception as e:
                    print(f"Error loading agent results for {model}: {e}")
            
            all_results[model] = model_results
        
        return all_results
    
    def generate_markdown_report(self, results: Dict[str, Dict[str, Any]], output_path: str):
        """Generate a markdown comparison report."""
        
        report_lines = [
            "# Local LLM Evaluation Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Overview",
            "",
            "This report compares the performance of multiple local LLMs on three key dimensions:",
            "1. **Code Generation** - HumanEval/MBPP-style coding tasks",
            "2. **Function Calling** - Tool use and API calling capabilities",
            "3. **Agent Capabilities** - Multi-step reasoning and planning",
            "",
            "## Models Evaluated",
            "",
        ]
        
        for model in self.models:
            report_lines.append(f"- {model}")
        
        report_lines.extend([
            "",
            "---",
            "",
            "## Summary Results",
            "",
            "### Overall Performance",
            "",
            "| Model | Code Gen | Tool Select | Params | Agent Acc | Reasoning |",
            "|-------|----------|-------------|--------|-----------|-----------|",
        ])
        
        # Build summary table
        for model in self.models:
            model_data = results.get(model, {})
            
            # Code generation accuracy
            code_gen = model_data.get("code_generation", {})
            code_acc = code_gen.get("accuracy", 0)
            code_str = f"{code_acc:.1%}" if code_acc else "N/A"
            
            # Function calling
            tool_data = model_data.get("function_calling", {})
            tool_acc = tool_data.get("tool_accuracy", 0)
            params_acc = tool_data.get("params_accuracy", 0)
            tool_str = f"{tool_acc:.1%}" if tool_acc else "N/A"
            params_str = f"{params_acc:.1%}" if params_acc else "N/A"
            
            # Agent
            agent_data = model_data.get("agent", {})
            agent_acc = agent_data.get("answer_accuracy", 0)
            reasoning = agent_data.get("avg_reasoning_score", 0)
            agent_str = f"{agent_acc:.1%}" if agent_acc else "N/A"
            reasoning_str = f"{reasoning:.1f}/3" if reasoning else "N/A"
            
            report_lines.append(
                f"| {model:20s} | {code_str:8s} | {tool_str:11s} | {params_str:6s} | {agent_str:9s} | {reasoning_str:9s} |"
            )
        
        report_lines.extend([
            "",
            "---",
            "",
            "## Detailed Results",
            "",
        ])
        
        # Detailed results for each model
        for model in self.models:
            model_data = results.get(model, {})
            safe_name = model.replace(":", "_").replace("/", "_")
            
            report_lines.extend([
                f"### {model}",
                "",
                "#### Code Generation",
                "",
            ])
            
            code_gen = model_data.get("code_generation", {})
            if code_gen:
                total = code_gen.get("total_tasks", 0)
                passed = code_gen.get("passed", 0)
                acc = code_gen.get("accuracy", 0)
                report_lines.extend([
                    f"- **Total Tasks:** {total}",
                    f"- **Passed:** {passed}",
                    f"- **Accuracy:** {acc:.2%}",
                    "",
                ])
            else:
                report_lines.append("- No results available\n")
            
            report_lines.extend([
                "#### Function Calling",
                "",
            ])
            
            tool_data = model_data.get("function_calling", {})
            if tool_data:
                total = tool_data.get("total_tasks", 0)
                correct_tools = tool_data.get("correct_tools", 0)
                correct_params = tool_data.get("correct_params", 0)
                tool_acc = tool_data.get("tool_accuracy", 0)
                params_acc = tool_data.get("params_accuracy", 0)
                report_lines.extend([
                    f"- **Total Tasks:** {total}",
                    f"- **Correct Tool Selection:** {correct_tools}/{total} ({tool_acc:.2%})",
                    f"- **Correct Parameters:** {correct_params}/{total} ({params_acc:.2%})",
                    "",
                ])
            else:
                report_lines.append("- No results available\n")
            
            report_lines.extend([
                "#### Agent Capabilities",
                "",
            ])
            
            agent_data = model_data.get("agent", {})
            if agent_data:
                total = agent_data.get("total_tasks", 0)
                correct = agent_data.get("correct_answers", 0)
                acc = agent_data.get("answer_accuracy", 0)
                reasoning = agent_data.get("avg_reasoning_score", 0)
                plan = agent_data.get("avg_plan_score", 0)
                report_lines.extend([
                    f"- **Total Tasks:** {total}",
                    f"- **Correct Answers:** {correct}/{total} ({acc:.2%})",
                    f"- **Avg Reasoning Score:** {reasoning:.2f}/3.0",
                    f"- **Avg Plan Score:** {plan:.2%}",
                    "",
                ])
            else:
                report_lines.append("- No results available\n")
            
            report_lines.append("---\n")
        
        # Rankings section
        report_lines.extend([
            "## Model Rankings",
            "",
            "### By Code Generation Accuracy",
            "",
        ])
        
        code_rankings = []
        for model in self.models:
            code_gen = results.get(model, {}).get("code_generation", {})
            acc = code_gen.get("accuracy", 0)
            code_rankings.append((model, acc))
        code_rankings.sort(key=lambda x: x[1], reverse=True)
        
        for i, (model, acc) in enumerate(code_rankings, 1):
            report_lines.append(f"{i}. **{model}**: {acc:.2%}")
        
        report_lines.extend([
            "",
            "### By Function Calling (Tool Selection)",
            "",
        ])
        
        tool_rankings = []
        for model in self.models:
            tool_data = results.get(model, {}).get("function_calling", {})
            acc = tool_data.get("tool_accuracy", 0)
            tool_rankings.append((model, acc))
        tool_rankings.sort(key=lambda x: x[1], reverse=True)
        
        for i, (model, acc) in enumerate(tool_rankings, 1):
            report_lines.append(f"{i}. **{model}**: {acc:.2%}")
        
        report_lines.extend([
            "",
            "### By Agent Capabilities (Answer Accuracy)",
            "",
        ])
        
        agent_rankings = []
        for model in self.models:
            agent_data = results.get(model, {}).get("agent", {})
            acc = agent_data.get("answer_accuracy", 0)
            agent_rankings.append((model, acc))
        agent_rankings.sort(key=lambda x: x[1], reverse=True)
        
        for i, (model, acc) in enumerate(agent_rankings, 1):
            report_lines.append(f"{i}. **{model}**: {acc:.2%}")
        
        report_lines.extend([
            "",
            "---",
            "",
            "## Conclusions",
            "",
            "Based on the evaluation results:",
            "",
        ])
        
        # Find best overall model
        overall_scores = {}
        for model in self.models:
            code_gen = results.get(model, {}).get("code_generation", {}).get("accuracy", 0)
            tool = results.get(model, {}).get("function_calling", {}).get("tool_accuracy", 0)
            agent = results.get(model, {}).get("agent", {}).get("answer_accuracy", 0)
            overall_scores[model] = (code_gen + tool + agent) / 3
        
        best_model = max(overall_scores.items(), key=lambda x: x[1])
        report_lines.append(f"- **Best Overall Model:** {best_model[0]} (avg score: {best_model[1]:.2%})")
        
        if code_rankings:
            report_lines.append(f"- **Best at Code Generation:** {code_rankings[0][0]}")
        if tool_rankings:
            report_lines.append(f"- **Best at Function Calling:** {tool_rankings[0][0]}")
        if agent_rankings:
            report_lines.append(f"- **Best at Agent Tasks:** {agent_rankings[0][0]}")
        
        report_lines.extend([
            "",
            "---",
            "",
            "*Report generated by Local LLM Evaluation Suite*",
        ])
        
        # Write report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\nMarkdown report saved to: {output_path}")
    
    def generate_json_summary(self, results: Dict[str, Dict[str, Any]], output_path: str):
        """Generate a JSON summary of all results."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "models": self.models,
            "results": results
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"JSON summary saved to: {output_path}")
    
    def run_all_evaluations(self, skip_code_gen: bool = False, 
                           skip_function_calling: bool = False,
                           skip_agent: bool = False):
        """Run all evaluations."""
        start_time = time.time()
        
        print("="*70)
        print("LOCAL LLM EVALUATION SUITE")
        print("="*70)
        print(f"\nModels to evaluate: {self.models}")
        print(f"Output directory: {self.output_dir}")
        print(f"\nStarting evaluations at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run evaluations
        if not skip_code_gen:
            self.run_code_generation_eval()
        
        if not skip_function_calling:
            self.run_function_calling_eval()
        
        if not skip_agent:
            self.run_agent_eval()
        
        # Collect and generate reports
        print("\n" + "="*70)
        print("GENERATING REPORTS")
        print("="*70)
        
        results = self.collect_results()
        
        # Generate markdown report
        md_path = os.path.join(self.output_dir, "evaluation_report.md")
        self.generate_markdown_report(results, md_path)
        
        # Generate JSON summary
        json_path = os.path.join(self.output_dir, "evaluation_summary.json")
        self.generate_json_summary(results, json_path)
        
        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"EVALUATION COMPLETE")
        print(f"{'='*70}")
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Reports saved to:")
        print(f"  - {md_path}")
        print(f"  - {json_path}")


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Local LLM Evaluation Suite Orchestrator")
    parser.add_argument("--models", nargs="+", 
                        default=["qwen3.6:27b", "qwen3-coder:30b", "deepseek-coder:33b", "qwen3-coder:latest"],
                        help="Models to evaluate")
    parser.add_argument("--output-dir", default="/root/local_coding_eval/results",
                        help="Output directory for results")
    parser.add_argument("--skip-code-gen", action="store_true",
                        help="Skip code generation evaluation")
    parser.add_argument("--skip-function-calling", action="store_true",
                        help="Skip function calling evaluation")
    parser.add_argument("--skip-agent", action="store_true",
                        help="Skip agent evaluation")
    args = parser.parse_args()
    
    orchestrator = EvaluationOrchestrator(args.models, args.output_dir)
    orchestrator.run_all_evaluations(
        skip_code_gen=args.skip_code_gen,
        skip_function_calling=args.skip_function_calling,
        skip_agent=args.skip_agent
    )


if __name__ == "__main__":
    main()
