# Library of Process Evaluation for Multi-Agent Systems (MAS Proc-Eval)


### Usage
Installing the client-server framework for MAS process evaluation:
```bash
pip install -e .
python -c "import mas_proceval"
```

Starting the Judge server: 
```bash
python -m mas_proceval.servers.server_judge
```

Using the client-server framework as a plug-in for the existing MAS systems:
```python
from mas_proceval import BaseClient, llm_parallel_search_decorator
client = BaseClient()

# this is the function we want to perform the search: 
@llm_parallel_search_decorator
def function():
    pass
```

### Available Judge Servers

**GPT-4o Standard Judge** (Recommended for general use):
```bash
python -m mas_proceval.servers.server_judge
```

**Process Reward Model (PRM)** (Qwen2.5-Math-PRM-7B):
```bash
python -m mas_proceval.servers.server_prm
```

**Reward Model (RM)** (Skywork-Reward-V2):
```bash
python -m mas_proceval.servers.server_rm
```

Judge servers should be started before launching any MAS experiments and will remain running to serve evaluation requests. You can run multiple judge servers simultaneously if needed.

---

## Process Verification by MAS System

Each MAS architecture has its own mechanism for enabling process verification. Below are detailed instructions for each system.

### Debate
**Agent-level verification** evaluates each individual debater's response:
```bash
python run_benchmarks.py --benchmark {dataset} --process-eval agent
```

**Round-level verification** evaluates complete debate rounds:
```bash
python run_benchmarks.py --benchmark {dataset} --process-eval round
```

---

### DyLAN
**For round-level (iteration-level) verification**, ensure your bash script uses the standard generation file:
```bash
python llmlp_gen_{dataset}.py  # Round/iteration level
```

Then run:
```bash
bash AIME/exp_aime_dylan.sh   # For AIME
bash GAIA/exp_gaia.sh         # For GAIA
```

**For agent-level verification**, modify your bash script to use the agent-call-level variant:
```bash
# In the bash script
python llmlp_gen_{dataset}_sub.py  # Agent call level
```

Then execute the same bash commands as above.


---
### MaAS
**Phase 1 - Optimization:**
```bash
python -m examples.maas.optimize --dataset {dataset} --round 1 --sample 4 --exec_model_name "gpt-5-mini"
```
**Phase 2 - Testing with Process Verification:**
**For agent-level verification:**
```bash
cp graph_sub.py graph.py
```

**For iteration-level verification:**
```bash
cp graph_iter.py graph.py
```

Then run the testing phase:
```bash
python -m examples.maas.optimize --dataset {dataset} --round 1 --sample 4 --exec_model_name "gpt-5-mini" --is_test True
```

The template files contain pre-configured decorators:
- `graph_sub.py`: Individual operators are decorated for fine-grained evaluation
- `graph_iter.py`: Complete workflow iterations are decorated for coarse-grained evaluation

---

### AFlow

AFlow requires manual configuration ONLY in testing phases. Process verification is enabled through a combination of configuration flags and decorator placement.

**Step 1 - Agent-level During Optimization:**

Edit `optimizer.py` and set:
```python
use_mas = False  # For Validation Phase
use_mas = True # For Testing Phase
```

Then run the optimization phase:
```bash
python run.py --dataset AIME24 --max_rounds 10 --validation_rounds 2 --opt_model_name gpt-5-mini --exec_model_name gpt-5-mini
```

**Step 2 - Iteration-level During Testing:**

After optimization completes and generates the workflow graph:

1. Set `use_mas = False` in `optimizer.py`
2. Manually add the `@llm_parallel_search_decorator` to appropriate methods in the generated `graph.py`
3. Refer to `MaAS/.../graph_iter.py` for examples of proper decorator placement

---

### ADAS
**For agent-level verification**, use the `search_sub.py` scripts:

**AIME24 or AIME25:**
```bash
python _aime/search_sub.py --dataset aime24 --expr_name aime24_results --n_generation 5
```

**GAIA:**
```bash
python _gaia/search_sub.py --expr_name gaia_results --n_generation 5
```

**For iteration-level verification**, use the `search_iter.py` scripts:

**AIME24 or AIME25:**
```bash
python _aime/search_iter.py --dataset aime24 --expr_name aime24_results --n_generation 5
```

**GAIA:**
```bash
python _gaia/search_iter.py --expr_name gaia_results --n_generation 5
```

---

### MAS-ZERO
**For agent-level verification**, modify the import to use the agent-call-level search function:
```python
from async_search_sub import search  # Agent call level
```

**For iteration-level verification**, modify the import to use the iteration-level search function:
```python
from async_search_iter import search  # Iteration level
```

After setting the appropriate import, run the planning phase:

**AIME24/25 Planning:**
```bash
python async_main_question.py --dataset workflow_search/aime24 --option plan --meta_model gpt-5-mini --node_model gpt-5-mini --blocks COT COT_SC Reflexion LLM_debate --n_generation 2 --save_dir test_iter_results
```

**GAIA Planning:**
```bash
python async_main_question.py --dataset workflow_search/gaia --option plan --meta_model gpt-5-mini --node_model gpt-5-nano --blocks COT COT_SC Reflexion LLM_debate WebSearch --n_generation 2 --save_dir test_results
```

**Oracle Verification**: For validating generated responses with an oracle judge:
```bash
python main_judge.py --dataset aime24 --judge_method oracle --baseline workflow_search --model gpt-5-mini --node_model gpt-5-mini --min_sample 0 --max_sample 29 --max_response_per_sample 5 --save_dir test_results
```
---

## Credits & Attribution

This repository extends and builds upon several foundational works in the field of Multi-Agent Systems (MAS). We are grateful to the authors of the following projects for open-sourcing their codebases:

* **[MAS-Zero](https://arxiv.org/abs/2505.14996)**
* **[MaaS](https://arxiv.org/abs/2502.04180)**
* **[AFlow](https://arxiv.org/abs/2410.10762)**
* **[DyLAN](https://arxiv.org/abs/2310.02170)**
* **[ADAS](https://arxiv.org/abs/2408.08435)**

Our work introduces MAS-ProVe, a systematic empirical study of process verification for MAS.

## Citation

If you use this codebase or the integrated architectures in your research, please cite our work and the original papers:

### MAS-ProVe
```bibtex
@article{}