# Local LLM Evaluation Report

**Generated:** 2026-05-06 (updated with corrected qwen3.6 scores)

## Overview

This report compares the performance of multiple local LLMs on three key dimensions:
1. **Code Generation** - HumanEval/MBPP-style coding tasks
2. **Function Calling** - Tool use and API calling capabilities
3. **Agent Capabilities** - Multi-step reasoning and planning

## Models Evaluated

- qwen3.6:27b
- qwen3.6:35b-a3b
- qwen3-coder:30b
- deepseek-coder:33b

---

## Summary Results

### Overall Performance

| Model | Code Gen | Tool Select | Params | Agent Acc | Reasoning |
|-------|----------|-------------|--------|-----------|-----------|
| qwen3.6:27b          | 80.0%    | 84.62%      | 84.62% | 100.0%    | 0.3/3     |
| qwen3.6:35b-a3b      | 70.0%    | 84.62%      | 84.62% | 100.0%    | 1.8/3     |
| qwen3-coder:30b      | 80.0%    | 76.92%      | 69.23% | 80.0%     | 2.8/3     |
| deepseek-coder:33b   | 90.0%    | 84.62%      | 69.23% | 10.0%     | 1.8/3     |

---

## Methodology Notes

### Timeout Configuration

Due to architectural differences between models, timeout values were adjusted to ensure fair evaluation:

- **qwen3.6 models (27b, 35b-a3b):** 1200s timeout per task
  - These are dense/MoE models that run slower on CPU-only inference
  - Extended timeout eliminates timeout artifacts and measures true capability

- **Other models (qwen3-coder, deepseek-coder):** 600s timeout per task
  - MoE architectures activate fewer parameters per forward pass
  - More efficient on CPU; 600s sufficient for all tasks

### Thinking Token Fix (qwen3.6 models)

qwen3.6 models use extended chain-of-thought reasoning, emitting `<think>...</think>` blocks
before generating actual output. Earlier evaluation runs used a 2048-token budget which was
exhausted by these reasoning blocks, leaving empty `generated_code` fields and causing
artificially low scores. The fix applied to all three eval scripts:

- **Strip `<think>...</think>` blocks** from raw model output before parsing
- **Increase `num_predict`**: 8192 for code generation and agent tasks, 4096 for function calling
- **Result**: qwen3.6:27b code generation improved from 40% → 80%; qwen3.6:35b-a3b from 50% → 70%

---

## Detailed Results

### qwen3.6:27b

#### Code Generation

- **Total Tasks:** 10
- **Passed:** 8
- **Accuracy:** 80.00%
- **Timeout:** 1200s (no timeouts occurred)
- **Note:** Previous 40% score was an artifact of thinking tokens exhausting the 2048 token budget

#### Function Calling

- **Total Tasks:** 13
- **Correct Tool Selection:** 11/13 (84.62%)
- **Correct Parameters:** 11/13 (84.62%)

#### Agent Capabilities

- **Total Tasks:** 10
- **Correct Answers:** 10/10 (100.00%)
- **Avg Reasoning Score:** 0.30/3.0
- **Avg Plan Score:** 10.00%

---

### qwen3.6:35b-a3b

#### Code Generation

- **Total Tasks:** 10
- **Passed:** 7
- **Accuracy:** 70.00%
- **Timeout:** 1200s (no timeouts occurred)
- **Note:** Previous 50% score was an artifact of thinking tokens exhausting the 2048 token budget

#### Function Calling

- **Total Tasks:** 13
- **Correct Tool Selection:** 11/13 (84.62%)
- **Correct Parameters:** 11/13 (84.62%)

#### Agent Capabilities

- **Total Tasks:** 10
- **Correct Answers:** 10/10 (100.00%)
- **Avg Reasoning Score:** 1.80/3.0
- **Avg Plan Score:** 70.00%

---

### qwen3-coder:30b

#### Code Generation

- **Total Tasks:** 10
- **Passed:** 8
- **Accuracy:** 80.00%

#### Function Calling

- **Total Tasks:** 13
- **Correct Tool Selection:** 10/13 (76.92%)
- **Correct Parameters:** 9/13 (69.23%)

#### Agent Capabilities

- **Total Tasks:** 10
- **Correct Answers:** 8/10 (80.00%)
- **Avg Reasoning Score:** 2.80/3.0
- **Avg Plan Score:** 92.17%

---

### deepseek-coder:33b

#### Code Generation

- **Total Tasks:** 10
- **Passed:** 9
- **Accuracy:** 90.00%

#### Function Calling

- **Total Tasks:** 13
- **Correct Tool Selection:** 11/13 (84.62%)
- **Correct Parameters:** 9/13 (69.23%)

#### Agent Capabilities

- **Total Tasks:** 10
- **Correct Answers:** 1/10 (10.00%)
- **Avg Reasoning Score:** 1.80/3.0
- **Avg Plan Score:** 14.17%

---

## Model Rankings

### By Code Generation Accuracy

1. **deepseek-coder:33b**: 90.00%
2. **qwen3.6:27b**: 80.00%
2. **qwen3-coder:30b**: 80.00%
4. **qwen3.6:35b-a3b**: 70.00%

### By Function Calling (Tool Selection Accuracy)

1. **qwen3.6:27b**: 84.62%
1. **qwen3.6:35b-a3b**: 84.62%
1. **deepseek-coder:33b**: 84.62%
4. **qwen3-coder:30b**: 76.92%

### By Agent Capabilities (Answer Accuracy)

1. **qwen3.6:27b**: 100.00%
1. **qwen3.6:35b-a3b**: 100.00%
3. **qwen3-coder:30b**: 80.00%
4. **deepseek-coder:33b**: 10.00%

---

## Conclusions

Based on the corrected evaluation results:

- **Best at Code Generation:** deepseek-coder:33b (90%)
- **Best at Function Calling:** qwen3.6:27b / qwen3.6:35b-a3b / deepseek-coder:33b (all 84.62%)
- **Best at Agent Tasks:** qwen3.6:27b / qwen3.6:35b-a3b (both 100%)
- **Best Overall Balance:** qwen3.6:27b (80% code gen, 84.62% tool calling, 100% agent)

### Key Insights

- **qwen3.6:27b** is the strongest all-rounder after fixing the thinking token issue — competitive on code generation (80%), top-tier on function calling and agent tasks
- **qwen3.6:35b-a3b** matches qwen3.6:27b on function calling and agent tasks, with better reasoning quality (1.8 vs 0.3 avg reasoning score), but slightly lower code generation (70%)
- **deepseek-coder:33b** remains the best pure code generator (90%) but is weak on multi-step agent reasoning (10%)
- **qwen3-coder:30b** offers the best reasoning quality score (2.8/3.0) with solid balanced performance
- The `<think>` token fix was critical: without it, qwen3.6 models appeared to underperform significantly due to token budget exhaustion, not actual capability gaps

---

*Report generated by Local LLM Evaluation Suite*
