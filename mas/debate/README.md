# LLM Debate Process Evaluation

A multi-agent debate system with process reward model (PRM) evaluation for mathematical reasoning and general question answering tasks.

## Overview

This system implements debate-based reasoning where multiple LLM agents collaborate and refine their solutions through iterative discussion. It supports various search strategies (greedy, beam search) and evaluation approaches (round-wise, agent-wise) with PRM-based scoring.

### Supported Benchmarks
- **AIME24/AIME25**: Mathematical reasoning problems
- **GAIA**: General AI Assistant benchmark

## Installation

```bash

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
