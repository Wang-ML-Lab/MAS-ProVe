import requests
from .server_base import BaseServer
from ..utils import read_config_yaml
import asyncio
import warnings

global_config = read_config_yaml()


class ServerPRM(BaseServer):
    def __init__(self,
                 host=None,
                 port=None,
                 model=None,
                 api_url=None,
                 max_parallel_calls=10):
        config = global_config.copy()
        host = host or config["server"].get("host", "localhost")
        port = port or config["server"].get("port", 8001)
        super().__init__(host, port)
        self.model = model or config["prm"]["model_path"]
        self.api_url = api_url or config["prm"]["api_url"]
        self.max_parallel_calls = max_parallel_calls
        self.semaphore = asyncio.Semaphore(max_parallel_calls)

    def format_candidate_for_eval(self, candidate):
        # For PRM, the candidate should be just the 'response'
        return "\n".join([candidate["context"], candidate["current-step"]])

    def _assemble_conversation_str(self, query, response):
        # Support Qwen2.5-Math-PRM model style only for now
        if "qwen2.5-math-prm" in self.model.lower():
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.model, trust_remote_code=True)
            system = "Please reason step by step, and put your final answer within \\boxed{}."
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": query},
                {"role": "assistant", "content": response + "<extra_0>"},
            ]
            conversation_str = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            return conversation_str
        else:
            raise ValueError(f"Unsupported model: {self.model}")

    def _post_http_request(self, prompt: dict):
        headers = {"User-Agent": "PRM-Server"}
        response = requests.post(self.api_url, headers=headers, json=prompt)
        return response

    def rewards2rankings(self, rewards):
        # Returns [0,1,2,...] sorted by reward descending (best first)
        indices = list(range(len(rewards)))
        return sorted(indices, key=lambda i: rewards[i], reverse=True)

    async def _process_request(self, request):
        assert request.get(
            "judge-type", None) == "prm", "Invalid judge type for PRM"
        task_type = request.get("task-type", None)
        candidates = request["partial_trajectories"]
        query = request.get("question", "")

        # Prepare candidate responses
        responses = [self.format_candidate_for_eval(candidate)
                     for candidate in candidates]
        conversation_strs = [self._assemble_conversation_str(
            query, response) for response in responses]
        prompts = [{"model": self.model, "input": conversation_str}
                   for conversation_str in conversation_strs]

        rewards = []
        async with self.semaphore:
            # Since requests is not async, run each prompt in thread pool
            import aiohttp

            async with aiohttp.ClientSession(headers={"User-Agent": "PRM-Server"}) as session:
                for prompt in prompts:
                    async with session.post(self.api_url, json=prompt) as response:
                        if response.status != 200:
                            text = await response.text()
                            raise ValueError(
                                f"Request failed with status code {response.status}: {text}"
                            )
                        try:
                            resp_json = await response.json()
                            # Assuming response.json()["data"][0]["data"][0][1] gives the reward for this prompt
                            reward = resp_json["data"][0]["data"][0][1]
                        except Exception as e:
                            text = await response.text()
                            raise ValueError(
                                f"Malformed response from PRM API: {text}"
                            ) from e
                        rewards.append(reward)

        assert len(rewards) == len(
            responses), "Rewards length mismatch with candidates"

        rankings = self.rewards2rankings(rewards)

        return {
            "rewards": rewards,
            "rankings": rankings
        }

    def process_request(self, request):
        return asyncio.run(self._process_request(request))


if __name__ == "__main__":
    # To run the PRM server in the shell, execute the following command:
    # CUDA_VISIBLE_DEVICES=0 vllm serve "/research/projects/mllab/public_llms/reward_models/qwen_rms/Qwen2.5-Math-PRM-7B" --task reward --tensor-parallel-size 1 --dtype bfloat16
    import subprocess
    import time

    # Command to start vllm PRM server
    vllm_command = [
        "CUDA_VISIBLE_DEVICES=0",
        "vllm",
        "serve",
        "/research/projects/mllab/public_llms/reward_models/qwen_rms/Qwen2.5-Math-PRM-7B",
        "--task", "reward",
        "--tensor-parallel-size", "1",
        "--dtype", "bfloat16"
    ]

    # Start the vllm server in the background
    print("Starting vllm (PRM) server in the background...")
    # Note: shell=True is required to parse environment assignment
    prm_proc = subprocess.Popen(
        " ".join(vllm_command),
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # Wait a few seconds for the PRM server to warm up
    print("Waiting 40 seconds for vllm server to start...")
    time.sleep(40)

    # Example main for manual testing
    prm_model_path = "/research/projects/mllab/public_llms/reward_models/qwen_rms/Qwen2.5-Math-PRM-7B"
    api_url = "http://localhost:8000/pooling"
    datas = [
        {"problem": "What is 2 + 2?", "response": "4"},
        {"problem": "What is 3 + 5?", "response": "8"}
    ]
    question = datas[0]["problem"]
    candidate_trajectories = [
        {"context": d["problem"], "current-step": d["response"]} for d in datas]
    request = {
        "judge-type": "prm",
        "task-type": "math",
        "partial_trajectories": candidate_trajectories,
        "question": question,
    }

    print("Starting ServerPRM...")
    server = ServerPRM(model=prm_model_path,
                       api_url=api_url, max_parallel_calls=2)
    server.start()
