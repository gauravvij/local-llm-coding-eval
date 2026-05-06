# Function Calling Evaluation - Verification Report

## Execution Summary
- **Date**: 2024-05-04
- **Script**: function_calling_eval.py
- **Timeout**: 600s per task (fixed from 300s)
- **Parser**: Raw JSON output support (not just markdown code blocks)

## Models Evaluated

| Model | Tool Accuracy | Params Accuracy | Tasks Passed |
|-------|---------------|-----------------|--------------|
| **qwen3.6:27b** | **84.62%** | **84.62%** | 11/13 |
| deepseek-coder:33b | 84.62% | 69.23% | 11/13 |
| qwen3-coder:30b | 76.92% | 69.23% | 10/13 |
| qwen3-coder:latest | 76.92% | 69.23% | 10/13 |

## Result Files Generated

All 4 result files successfully created in `/root/local_coding_eval/results/`:

1. ✅ `tool_calling_qwen3.6_27b.json` (9,349 bytes, 244 lines)
2. ✅ `tool_calling_qwen3-coder_30b.json` (9,906 bytes, 243 lines)
3. ✅ `tool_calling_deepseek-coder_33b.json` (9,521 bytes, 245 lines)
4. ✅ `tool_calling_qwen3-coder_latest.json` (9,946 bytes, 243 lines)

## Parser Fix Verification

✅ **Raw JSON parsing working correctly**
- Models outputting JSON without markdown fences are parsed successfully
- Multiple JSON object patterns detected and extracted
- No timeout errors with 600s timeout setting

## Task Breakdown

**Simple Tasks (10 tasks)**: All models performed well on single tool calls
**Complex Tasks (3 tasks)**: Multi-step scenarios where models sometimes output multiple tools or conversational responses

## Key Findings

1. **qwen3.6:27b** achieved highest parameter accuracy (84.62%) - best at extracting exact parameter values
2. **deepseek-coder:33b** tied for tool selection accuracy (84.62%) but lower parameter accuracy
3. **qwen3-coder** variants (30b and latest) showed consistent performance around 77% tool accuracy
4. Complex multi-step tasks remain challenging for all models

## Verification Status

✅ All deliverables complete
✅ JSON files valid and parseable
✅ Parser fix working for raw JSON output
✅ Timeout increased to 600s (no timeouts occurred)
