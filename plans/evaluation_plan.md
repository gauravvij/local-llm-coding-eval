# Local Coding Agent LLM Evaluation Plan

## Goal
Evaluate multiple local LLMs (Qwen3.6-27B, Qwen3-Coder-30B-A3B, DeepSeek-Coder) for coding agent capabilities on CPU-only infrastructure.

## Research Summary
- **EvalPlus v0.3.1**: Enhanced HumanEval + MBPP with more test cases
- **BFCL (Berkeley Function Calling Leaderboard)**: Standard for tool-calling evaluation
- **bigcode-evaluation-harness**: Comprehensive coding benchmark suite
- **Terminal-Bench**: Agent-specific coding benchmark (used in Qwen3.6 paper)

## Evaluation Dimensions

### 1. Code Generation (EvalPlus)
- HumanEval: 164 Python programming problems
- MBPP: 500+ Python problems with test cases
- Metrics: pass@1, pass@10, pass@100

### 2. Function Calling / Tool Use (BFCL)
- Simple function calls
- Multiple function calls
- Parallel function calls
- Function calling with relevance detection
- Metrics: accuracy per category

### 3. Agent Capabilities (Custom)
- Multi-step reasoning
- Code editing/debugging
- File system operations
- Tool chaining

## Models to Evaluate
1. **Qwen3.6-27B** (dense, best benchmark scores)
2. **Qwen3-Coder-30B-A3B** (MoE, proven stable)
3. **DeepSeek-Coder-33B** (strong alternative)
4. **Qwen3-Coder-8B** (baseline/smaller option)

## Infrastructure Setup
- Ollama for model serving
- Python evaluation harness
- CPU-optimized inference (32 cores, 125GB RAM)

## Deliverables
| File | Description |
|------|-------------|
| `/root/local_coding_eval/setup_ollama.sh` | Ollama installation script |
| `/root/local_coding_eval/pull_models.sh` | Model download script |
| `/root/local_coding_eval/eval_code_generation.py` | EvalPlus-based evaluation |
| `/root/local_coding_eval/eval_function_calling.py` | BFCL-style evaluation |
| `/root/local_coding_eval/eval_agent.py` | Agent task evaluation |
| `/root/local_coding_eval/run_full_eval.py` | Orchestrator script |
| `/root/local_coding_eval/results/` | Evaluation results |
| `/root/local_coding_eval/report.md` | Final comparison report |

## Success Criteria
- All models successfully loaded and inference working
- At least 2 benchmark dimensions evaluated per model
- Quantitative comparison table generated
- Recommendation for best coding agent model

## Notes
- CPU-only inference will be slower but sufficient for evaluation
- May need to sample subsets for faster iteration
- Focus on pass@1 (single attempt) as most realistic for agents
