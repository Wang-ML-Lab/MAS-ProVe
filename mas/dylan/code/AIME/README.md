# AIME Implementation for DyLAN Framework

## Overview

This folder contains the implementation for testing DyLAN on AIME (American Invitational Mathematics Examination) problems from 2024 and 2025.

## Key Differences from MATH Benchmark

| Aspect | MATH | AIME |
|--------|------|------|
| **Difficulty** | High school competition | Top 5% of AMC scorers |
| **Answer Format** | Various (equations, numbers) | Integer 0-999 only |
| **Problem Complexity** | Moderate | Very high |
| **Reasoning Depth** | Multi-step | Deep multi-step |
| **Recommended Model** | GPT-3.5-turbo acceptable | GPT-4 recommended |
| **Temperature** | 0.2 | 0.3 (more exploration) |
| **Max Tokens** | 2048 | 3072 (longer reasoning) |

## Files

### Core Implementation
- **`llmlp_gen_aime.py`**: Main experiment script
  - Processes AIME problems with 4-agent, 3-round collaboration
  - Includes AIME-specific examples and prompts
  - Early stopping with 2/3 consensus mechanism
  - Ranking step to select top 2 solutions

- **`eval_aime.py`**: Evaluation script
  - Computes accuracy overall and by year
  - Tracks correct/incorrect problems
  - Reports API usage statistics

- **`exp_aime.sh`**: Batch processing script
  - Processes all problems in dataset
  - Handles multiple years
  - Batches of 15 problems (AIME contest size)

### Data Setup
- **`download_aime_dataset.py`**: Dataset preparation
  - Creates sample problem for testing
  - Instructions for manual dataset creation
  - Expected JSON format specification

## Dataset Structure

```
/common/home/vv382/Datasets/AIME/
├── 2024/
│   ├── 0.json   (AIME 2024 Problem 1)
│   ├── 1.json   (AIME 2024 Problem 2)
│   ├── ...
│   └── 14.json  (AIME 2024 Problem 15)
├── 2025/
│   ├── 0.json   (AIME 2025 I Problem 1)
│   ├── ...
│   └── 14.json
└── sample/
    └── 0.json   (Test problem)
```

### JSON Format

Each problem file contains:

```json
{
  "problem": "Problem text with LaTeX formatting...",
  "solution": "Step-by-step solution ending with \\boxed{answer}",
  "answer": "123",
  "year": "2024",
  "problem_number": 1
}
```

**Important**: 
- Answer must be an integer 0-999
- Solution should include `\boxed{answer}` for extraction
- Problem numbers are 1-15, but filenames are 0-indexed (0.json = Problem 1)

## Setup Instructions

### 1. Obtain AIME Problems

AIME problems can be obtained from:
- **Official**: [MAA Math Competitions](https://maa.org/math-competitions)
- **Art of Problem Solving**: [AIME Wiki](https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions)

### 2. Prepare Dataset

```bash
# Run the dataset preparation script
python /common/home/vv382/DyLAN-PRM/download_aime_dataset.py

# Or manually create JSON files for each problem
# Follow the format shown above
```

### 3. Test with Sample Problem

```bash
cd /common/home/vv382/DyLAN-PRM/code/AIME
export OPENAI_API_KEY='your-api-key-here'

# Test on sample problem
python llmlp_gen_aime.py \
  /common/home/vv382/Datasets/AIME/sample \
  0 0 gpt-4 gpt-4

# Evaluate results
python eval_aime.py llmlp_aime_gpt-4 None
```

### 4. Run Full Experiments

```bash
cd /common/home/vv382/DyLAN-PRM/code/AIME
export OPENAI_API_KEY='your-api-key-here'

# Process all AIME problems
bash exp_aime.sh

# Or run specific year
python llmlp_gen_aime.py \
  /common/home/vv382/Datasets/AIME/2024 \
  0 14 gpt-4 gpt-4
```

## Architecture Details

### Agent Configuration
- **Number of Agents**: 4
- **Rounds**: 3
- **Model**: GPT-4 (GPT-3.5-turbo also supported but not recommended)
- **Temperature**: 0.3 (higher than MATH for more exploration)
- **Max Tokens**: 3072 (more than MATH for complex reasoning)

### Collaboration Flow

```
Round 0: Independent Solving
├── Agent 1 → solution₁
├── Agent 2 → solution₂
├── Agent 3 → solution₃
└── Agent 4 → solution₄
     └── Check consensus (2/3 majority)
          ├── YES → Stop
          └── NO → Continue

Round 1: Cross-Pollination
├── Agents see all other solutions
├── Each agent revises their answer
└── Check consensus
     ├── YES → Stop
     └── NO → Continue

Ranking Step:
└── Judge ranks all 4 solutions → Select top 2

Round 2: Final Refinement
├── Top 2 agents revise with each other's input
└── Output final answer (majority vote)
```

### Consensus Mechanism

- **Threshold**: > 2/3 of active agents must agree
- **Comparison**: Uses `is_equiv()` for mathematical equivalence
- **Early Stopping**: Can stop after Round 0 or Round 1 if consensus reached
- **Efficiency**: Reduces API calls by ~30% on easier problems

## AIME-Specific Adaptations

### 1. Examples
Uses real AIME problems as few-shot examples instead of simple GSM8K problems:
- Divisibility and number theory
- Combinatorics with generating functions
- Complex geometric progressions

### 2. System Prompt
```
"You are an expert mathematician solving AIME problems. 
Explain your reasoning thoroughly at each step. 
AIME answers are always integers from 0 to 999."
```

### 3. Answer Extraction
- Looks for `\boxed{integer}`
- Validates answer is 0-999
- Strips formatting and converts to canonical form

### 4. Prompting Strategy
- Emphasizes mathematical rigor
- Encourages critical thinking about peer solutions
- Reminds agents that answers must be integers

## Expected Performance

### Baseline Expectations (GPT-4)
- **Human Expert**: ~12/15 correct (~80%)
- **Top AI Systems (2024)**: ~8-10/15 correct (~60%)
- **DyLAN Target**: ~9-11/15 correct (~65-75%)

### Cost Estimates (per problem)
- **GPT-4**: $0.30-0.60 per problem
- **15 problems**: $4.50-9.00 per contest
- **Full 2024+2025**: ~$18-36 for all 60 problems

### Runtime
- **Per Problem**: 2-5 minutes (with 4 agents, 3 rounds)
- **Per Contest (15)**: ~45-90 minutes
- **Parallel**: Can run multiple problems simultaneously

## Evaluation Metrics

The evaluation script reports:

1. **Overall Accuracy**: Percentage correct across all problems
2. **By Year**: Accuracy for 2024, 2025 I, 2025 II separately
3. **Problem-by-Problem**: Shows which specific problems failed
4. **API Usage**: Total responses and average per problem
5. **Consensus Rate**: How often early stopping occurred

## Troubleshooting

### Issue: "No answer found"
- **Cause**: Model didn't use `\boxed{}` format
- **Solution**: Check EXAMPLES in script have proper formatting

### Issue: Wrong answers on number theory problems
- **Cause**: AIME number theory is very hard
- **Solution**: Consider adding more domain-specific examples

### Issue: High API costs
- **Cause**: Every problem uses 8-12+ calls with GPT-4
- **Solution**: 
  - Start with sample problems
  - Use GPT-3.5-turbo for initial testing
  - Implement better early stopping

### Issue: Timeout errors
- **Cause**: Complex problems take a long time
- **Solution**: Increase max_tokens or split into smaller batches

## Comparison with Other Benchmarks

| Feature | MATH | MMLU | HumanEval | AIME |
|---------|------|------|-----------|------|
| Architecture | Direct script | LLMLP class | CoLLMLP | Direct script |
| Consensus | 2/3 majority | Most frequent | BLEU + exec | 2/3 majority |
| Validation | `is_equiv()` | String match | Code run | `is_equiv()` |
| Difficulty | Medium | Easy-Medium | Medium | Very Hard |
| Best Model | GPT-3.5 OK | GPT-3.5 OK | GPT-4 | GPT-4 required |

## Future Improvements

1. **Better Examples**: Add more AIME-specific examples
2. **Specialized Agents**: Use role-based agents (Number Theory Expert, Geometer, etc.)
3. **Longer Context**: Use models with longer context for complex problems
4. **Verification**: Add symbolic verification step for answers
5. **Adaptive Rounds**: Increase rounds for harder problems

## Citation

If you use this AIME implementation, please cite the original DyLAN paper:

```bibtex
@misc{liu2023dynamic,
    title={Dynamic LLM-Agent Network: An LLM-agent Collaboration Framework with Agent Team Optimization},
    author={Zijun Liu and Yanzhe Zhang and Peng Li and Yang Liu and Diyi Yang},
    year={2023},
    eprint={2310.02170},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
