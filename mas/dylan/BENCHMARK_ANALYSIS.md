# DyLAN Framework Analysis - Three Benchmarks

## Overview of Architecture

DyLAN uses a **Multi-Agent Neural Network** metaphor where:
- **Nodes (Neurons)**: LLM agents with specific roles
- **Edges**: Information flow between agents across rounds
- **Activation**: Ranking/selection mechanism (listwise ranking)
- **Forward Pass**: Multi-round agent collaboration
- **Backward Pass**: Agent importance score calculation

---

## 1. MATH Benchmark

### Structure
- **Input Format**: JSON files with `problem`, `solution`, `level`, `type`
- **Directory**: `<dataset>/test/<problem_type>/<number>.json`
- **Problem Types**: Algebra, Geometry, Number Theory, etc.

### Key Components

#### Main Script: `llmlp_gen_math_listwise_deeper_markov.py`
- **Arguments**: `SUB_DIR MIN_FILE MAX_FILE MODEL ENGINE`
- **Flow**: Direct agent-to-agent collaboration (no separate LLMLP class)
- **Architecture**: 4 agents, 3 rounds
- **Consensus Mechanism**: Early stopping when 2/3 agents agree

#### Process Flow:
1. **Round 0**: All 4 agents independently solve the problem
   - Check consensus after each agent (early stop if ≥3 agree)
2. **Round 1**: Agents see other solutions and revise
   - Can early stop if consensus reached
3. **Ranking**: If no consensus, rank best 2 solutions
4. **Round 2**: Top 2 agents revise with cross-information

#### Answer Extraction:
- **Method**: `extract_math_answer()` - extracts from `\\boxed{}` format
- **Comparison**: `is_equiv()` - mathematical equivalence checking
- **Examples**: Simple arithmetic examples (GSM8K-style)

#### Evaluation (`eval_math.py`):
- Reads JSON results from output directory
- Groups by problem type and level
- Computes accuracy per category
- Reports total accuracy and statistics

---

## 2. MMLU Benchmark

### Structure
- **Input Format**: CSV files (question, A, B, C, D, answer)
- **Problem Type**: Multiple choice (A/B/C/D)
- **Subjects**: Various academic subjects

### Key Components

#### Main Script: `llmlp_listwise_mmlu.py`
- **Arguments**: `QUERY_CSV EXP_NAME MODEL DIR_NAME ROLES`
- **Architecture**: Uses `LLMLP` class (proper neural network structure)
- **Agent Roles**: Configurable (e.g., Economist, Doctor, Lawyer, Mathematician)

#### LLMLP Class (`LLMLP.py`):
```python
- agents: Number of agents per round
- rounds: Number of collaboration rounds (default 3)
- activation: "listwise" ranking
- qtype: "single_choice"
```

#### Process Flow:
1. **Initialization**: Create agent neurons and edges
2. **Forward Pass**: 
   - Round 0: Independent answers
   - Round 1+: Cross-pollination with ranking
3. **Backward Pass**: Calculate agent importance scores
4. **Output**: Accuracy, response counts, importance scores

#### Answer Extraction:
- **Method**: `parse_single_choice()` - extracts A/B/C/D
- **Comparison**: String equality

---

## 3. HumanEval Benchmark

### Structure
- **Input Format**: Python code completion tasks
- **Source**: `human-eval` package (read_problems())
- **Fields**: `task_id`, `prompt`, `entry_point`, `test`

### Key Components

#### Main Script: `llmlp_listwise_human_eval.py`
- **Arguments**: `PART EXP_NAME MODEL DIR_NAME ROLES JUDGES`
- **Architecture**: Uses `CoLLMLP` class (Co-Laboration LLMLP with judges)
- **Special Feature**: Separate judge agents to evaluate code

#### CoLLMLP Class (`CoLLMLP.py`):
```python
- agents: Code generation agents
- judges: Code evaluation agents  
- rounds: Number of collaboration rounds
- qtype: "code_completion"
```

#### Process Flow:
1. **Round 0**: Agents generate code solutions
2. **Judgment Round**: Judge agents evaluate and critique code
3. **Refinement**: Agents revise based on judge feedback
4. **Execution**: Run generated code against test cases
5. **Evaluation**: Use `human-eval` metrics (pass@k)

#### Answer Extraction:
- **Method**: `parse_code_completion()` - extracts Python code blocks
- **Comparison**: BLEU score (≥90% similarity threshold)
- **Validation**: Actual code execution with test cases

#### Unique Features:
- **Test Generation**: Judges can suggest test cases
- **Code Execution**: `check_function_result()` runs code safely
- **Multi-Stage**: Generation → Judgment → Refinement

---

## Common Patterns Across All Three

### 1. **Multi-Round Collaboration**
- Round 0: Independent generation
- Round 1+: Cross-pollination with peer solutions
- Early stopping: Consensus mechanism

### 2. **Agent Roles**
- Domain-specific roles (Mathematician, Programmer, etc.)
- Role-based prompting affects answer quality

### 3. **Ranking/Selection**
- Listwise ranking to select top solutions
- Reduces agents from N to 2 for final rounds

### 4. **Output Format**
```json
{
  "completions": [...],  // All agent responses per round
  "answer": "...",       // Ground truth
  "result": "...",       // Final prediction
  "importance": [...]    // Agent importance scores
}
```

### 5. **Evaluation Metrics**
- **MATH**: Accuracy per problem type/level
- **MMLU**: Accuracy per subject
- **HumanEval**: Pass@k (functional correctness)

---

## Key Differences

| Feature | MATH | MMLU | HumanEval |
|---------|------|------|-----------|
| Input | JSON files | CSV files | Python dict |
| Answer Type | Free-form math | Multiple choice | Code |
| Architecture | Direct script | LLMLP class | CoLLMLP class |
| Agents | 4 fixed | Configurable | Agents + Judges |
| Validation | Symbolic equiv | String match | Code execution |
| Consensus | 2/3 majority | Most frequent | BLEU + tests |

---

## For AIME Implementation

You'll need to decide:

1. **Input Format**: JSON (like MATH) or custom?
2. **Answer Type**: Integer (0-999) - similar to MATH
3. **Architecture**: Use MATH's direct approach or LLMLP class?
4. **Validation**: Exact integer match
5. **Difficulty**: AIME is harder - may need more rounds/agents
6. **Examples**: AIME-specific examples (not GSM8K)

Recommendation: **Start with MATH structure** since AIME is also mathematics, but:
- Update examples to AIME-level problems
- Adjust temperature/max_tokens for more complex reasoning
- Consider increasing rounds (3→4 or 5)
- Add AIME-specific answer extraction (integers 0-999)
