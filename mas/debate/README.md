# LLM Debate Process Evaluation

A multi-agent debate system with process reward model (PRM) evaluation for mathematical reasoning and general question answering tasks.

## Overview

This system implements debate-based reasoning where multiple LLM agents collaborate and refine their solutions through iterative discussion. It supports various search strategies (greedy, beam search) and evaluation approaches (round-wise, agent-wise) with PRM-based scoring.

### Supported Benchmarks
- **AIME24/AIME25**: Mathematical reasoning problems
- **GAIA**: General AI Assistant benchmark

## Installation

```bash
# Install required packages
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

## Quick Start

### Basic Debate (No PRM)

Run a simple multi-agent debate on AIME24:

```bash
python run_benchmarks.py \
  --benchmark aime24 \
  --model gpt-5-mini \
  --num-agents 3 \
  --num-rounds 2 \
  --max-aime-examples 5
```

### With Search Strategies

#### RoundWise Greedy Search

```bash
python run_benchmarks.py \
  --benchmark aime24 \
  --evaluation-type greedy \
  --model gpt-5-mini \
  --num-candidates 3 \
  --greedy-rounds 2 \
  --judge-type scoring \
  --max-aime-examples 10
```

#### RoundWise Beam Search

```bash
python run_benchmarks.py \
  --benchmark aime24 \
  --evaluation-type beam \
  --model gpt-5-mini \
  --beam-width 3 \
  --beam-rounds 2 \
  --judge-type scoring \
  --max-aime-examples 10
```

#### AgentWise Greedy Search

```bash
python run_benchmarks.py \
  --benchmark aime24 \
  --evaluation-type agentwise-greedy \
  --model gpt-5-mini \
  --num-candidates 3 \
  --greedy-rounds 2 \
  --judge-type scoring \
  --max-aime-examples 10
```

#### AgentWise Beam Search

```bash
python run_benchmarks.py \
  --benchmark aime24 \
  --evaluation-type agentwise-beam \
  --model gpt-5-mini \
  --beam-width 3 \
  --beam-rounds 2 \
  --judge-type scoring \
  --max-aime-examples 10
```

### With PRM Evaluation

```bash
python run_benchmarks.py \
  --benchmark aime24 \
  --evaluation-type greedy \
  --model gpt-5-mini \
  --judge-type prm \
  --prm-model-path /path/to/prm/model \
  --prm-api-url http://localhost:8000/v1 \
  --num-candidates 3 \
  --max-aime-examples 10
```

### With Web Search (GAIA)

```bash
python run_benchmarks.py \
  --benchmark gaia \
  --model gpt-5-mini \
  --num-agents 3 \
  --num-rounds 2 \
  --use-tools \
  --max-gaia-examples 20
```

## Configuration Options

### Model Configuration
- `--model`: Model to use (default: `gpt-5-mini`)
- `--num-agents`: Number of debate agents (default: 2-3)
- `--num-rounds`: Number of debate rounds (default: 2-3)

### Search Strategy
- `--evaluation-type`: Type of evaluation
  - `none`: Basic debate without search
  - `greedy`: RoundWise greedy search
  - `beam`: RoundWise beam search
  - `agentwise-greedy`: AgentWise greedy search
  - `agentwise-beam`: AgentWise beam search

### Greedy Search Parameters
- `--num-candidates`: Number of candidate solutions (default: 3)
- `--greedy-rounds`: Number of greedy search rounds (default: 2)

### Beam Search Parameters
- `--beam-width`: Beam width for search (default: 3)
- `--beam-rounds`: Number of beam search rounds (default: 2)

### Judge/PRM Configuration
- `--judge-type`: Judge scoring type (default: `scoring`)
  - `scoring`: Standard scoring judge
  - `prm`: Process reward model
  - `ranking`: Ranking-based judge
- `--judge-model`: Model for judging (defaults to same as `--model`)
- `--prm-model-path`: Path to PRM model weights (for `prm` judge type)
- `--prm-api-url`: API URL for PRM server (for `prm` judge type)

### Other Options
- `--use-tools`: Enable web search capability (GAIA)
- `--enable-summarization`: Enable response summarization
- `--output-dir`: Directory for results (default: `results`)
- `--max-aime-examples`: Max AIME problems to evaluate
- `--aime-problem-ids`: Specific AIME problem IDs (e.g., `--aime-problem-ids 1 3 5`)