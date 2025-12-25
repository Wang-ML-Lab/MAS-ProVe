### Starting a vLLM PRM server
```bash
vllm serve "<path-to-prm-model>" --task reward
```

### Quick Demo for Accessing the Server
```python
from prm_vllm_server import request_for_process_rewards

# Example usage
prm_model_path = "/research/projects/mllab/public_llms/reward_models/qwen_rms/Qwen2.5-Math-PRM-7B"

datas = [
    {"problem": "What is 2 + 2?", "response": "4"},
    {"problem": "What is 3 + 5?", "response": "8"}
]

# Load the model and tokenizer
rewards = request_for_process_rewards(
    query=datas[0]["problem"],
    step_responses=[data["response"] for data in datas],
    model=prm_model_path,
    api_url="http://localhost:8000/pooling"
)

print(rewards)
```