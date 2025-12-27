import requests
from .server_base import BaseServer
from ..utils import read_config_yaml
import asyncio
import warnings

global_config = read_config_yaml()
from litellm import acompletion

SYSTEM_PROMPT = """
You are a summarization assistant for multi-agent system processes.

Your task is to summarize the given context into a concise and informative summary.
The total length of the summary should be fewer than {max_context_length} tokens.

IMPORTANT PRINCIPLES:
- This is NOT style normalization.
- This is NOT reformatting to a fixed template.
- The cleaned solution should look like the SAME author wrote it,
  just more concisely.

STRICT REQUIREMENTS:
1) Faithfulness:
   - Do NOT change the meaning.
   - Do NOT change the final answer(s).
2) Reasoning preservation:
   - Keep ALL logical reasoning steps and ALL necessary mathematical derivations.
   - You may remove repeated steps, detours, or restatements ONLY if the reasoning remains intact.
3) Language & style preservation:
   - Preserve the original narrative tone (e.g. exploratory, explanatory, corrective).
   - Preserve structural choices used in the original solution
     (such as step labels, section headers, or paragraph structure),
     unless they are clearly redundant.
4) Formatting:
   - Do NOT convert narrative explanations into bullet fragments or terse notes.
   - Do NOT collapse structured explanations into formula-only derivations.
   - Keep LaTeX for mathematics.

Output ONLY the cleaned solution text. No extra commentary.
"""

class ServerRM(BaseServer):
    def __init__(self,
                 host=None,
                 port=None,
                 model=None,
                 api_url="http://localhost:8000/classify",
                 max_parallel_calls=50,
                 summary_mode="enforce",
                 summary_model="gpt-5-mini",
                 max_context_length=16384):
        config = global_config.copy()
        # print(f"[DEBUG] Initializing ServerRM with config: {config}")
        host = host or config["server"].get("host", "localhost")
        port = port or config["server"].get("port", 8001)
        super().__init__(host, port)
        self.model = model or config["prm"]["model_path"]
        self.api_url = api_url or config["prm"]["api_url"]
        self.max_parallel_calls = max_parallel_calls
        self._semaphore = None  # Will be created per event loop

        # summarization should be performed when the context is too long for the PRM model.
        assert summary_mode in ["enforce", "optional"], "Invalid summary mode"
        self.summary_mode = summary_mode
        self.max_context_length = max_context_length
        self.openai_api_key = global_config["api_keys"]["openai"]
        self.summary_model_name = summary_model

    async def _get_semaphore(self):
        # Ensure semaphore is created per event loop
        if self._semaphore is None or self._semaphore._loop != asyncio.get_running_loop():
            self._semaphore = asyncio.Semaphore(self.max_parallel_calls)
        return self._semaphore
    
    async def summarize_context(self, context: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(max_context_length=self.max_context_length)},
            {"role": "user", "content": f"Please summarize the given context into a concise and informative summary. The total length of the summary should be fewer than 4096 tokens.\n\nContext: {context}"},
        ]
        # print(f"[DEBUG] Summarize context called with context length: {len(context)}")
        semaphore = await self._get_semaphore()
        async with semaphore:
            response = await acompletion(
                model=self.summary_model_name,
                messages=messages,
                api_key=self.openai_api_key,
                reasoning_effort="minimal",
                verbosity="low",
            )
        print(f"[DEBUG] Summarization response received.")
        return response.choices[0].message.content

    def format_candidate_for_eval(self, candidate):
        # For PRM, the candidate should be just the 'response'
        context = candidate["context"] if isinstance(candidate["context"], str) else str(candidate["context"])
        current_step = candidate["current-step"] if isinstance(candidate["current-step"], str) else str(candidate["current-step"])
        # print(f"[DEBUG] Formatting candidate for eval: context={context[:50]}, current_step={current_step[:50]}")
        return "\n".join([context, current_step])

    def _assemble_conversation_str(self, query, response):
        # Support Qwen2.5-Math-PRM model style only for now
        if "skywork-reward-v2-llama-3.1-8b" in self.model.lower():
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.model, trust_remote_code=True)
            # print(f"[DEBUG] Assembling conversation string for model: {self.model}")
            # from the skywork's huggingface repo:
            messages = [
                {"role": "user", "content": query}, 
                {"role": "assistant", "content": response}
            ]
            conversation_str = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )
            if tokenizer.bos_token is not None \
                and conversation_str.startswith(tokenizer.bos_token):
                conversation_str = conversation_str[len(tokenizer.bos_token):]
            # print(f"[DEBUG] Conversation string: {conversation_str[:100]}")
            return conversation_str
        else:
            raise ValueError(f"Unsupported model: {self.model}")

    def _post_http_request(self, prompt: dict):
        headers = {"User-Agent": "RM-Server"}
        # print(f"[DEBUG] Posting HTTP request to {self.api_url} with prompt: {prompt}")
        response = requests.post(self.api_url, headers=headers, json=prompt)
        # print(f"[DEBUG] HTTP response status: {response.status_code}")
        return response

    def rewards2rankings(self, rewards):
        # Returns [0,1,2,...] sorted by reward descending (best first)
        indices = list(range(len(rewards)))
        # print(f"[DEBUG] Rewards: {rewards}")
        return sorted(indices, key=lambda i: rewards[i], reverse=True)

    async def _process_request(self, request):
        # assert request.get(
        #     "judge-type", None) == "prm", "Invalid judge type for PRM"
        task_type = request.get("task-type", None)
        candidates = request["partial_trajectories"]
        query = request.get("question", "")

        # print(f"[DEBUG] Processing request: task_type={task_type}, num_candidates={len(candidates)}")

        # Prepare candidate responses
        responses = [self.format_candidate_for_eval(candidate)
                     for candidate in candidates]

        # summarize the context;
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model, trust_remote_code=True)
        if self.summary_mode == "enforce" or \
            any(len(tokenizer.encode(response)) > self.max_context_length for response in responses):
            tasks = [self.summarize_context(response) for response in responses]
            summarized_responses = await asyncio.gather(*tasks)
            responses = summarized_responses
        # print(f"[DEBUG] Summarized responses: {responses}")

        conversation_strs = [self._assemble_conversation_str(
            query, response) for response in responses]
        # print(f"[DEBUG] Conversation strings: {[s[:100] for s in conversation_strs]}")
        
        prompts = [{
            "model": self.model, 
            "input": conversation_str, 
            "activation": False, # should not apply the activation function to match the real reward values.
        } for conversation_str in conversation_strs]

        # print(f"[DEBUG] Prompts: {prompts}")

        rewards = []
        semaphore = await self._get_semaphore()
        async with semaphore:
            # Since requests is not async, run each prompt in thread pool
            import aiohttp

            async with aiohttp.ClientSession(headers={"User-Agent": "PRM-Server"}) as session:
                for prompt in prompts:
                    # print(f"[DEBUG] Sending prompt to RM API: {prompt}")
                    async with session.post(self.api_url, json=prompt) as response:
                        # print(f"[DEBUG] RM API response status: {response.status}")
                        if response.status != 200:
                            text = await response.text()
                            warnings.warn(
                                f"Request failed with status code {response.status}: {text}; assigning a reward of 0.0"
                            )
                            reward = -float('inf')
                        try:
                            resp_json = await response.json()
                            # print(f"[DEBUG] RM API response JSON: {resp_json}")
                            # Assuming response.json()["data"][0]["probs"][0] gives the reward for this prompt
                            reward = resp_json["data"][0]["probs"][0]
                        except Exception as e:
                            text = await response.text()
                            # print(f"[DEBUG] Exception parsing RM API response: {e}, text: {text}")
                            raise ValueError(
                                f"Malformed response from RM API: {text}"
                            ) from e
                        rewards.append(reward)

        assert len(rewards) == len(
            responses), "Rewards length mismatch with candidates"

        print(f"[DEBUG] Final rewards: {rewards}")
        rankings = self.rewards2rankings(rewards)

        # print(f"[DEBUG] Rankings: {rankings}")
        return {
            "rewards": rewards,
            "rankings": rankings
        }

    def process_request(self, request):
        return asyncio.run(self._process_request(request))


if __name__ == "__main__":
    # To run the PRM server in the shell, execute the following command:
    # CUDA_VISIBLE_DEVICES=0 vllm serve "/research/projects/mllab/public_llms/reward_models/Skywork-Reward-V2-Llama-3.1-8B" --task classify --tensor-parallel-size 1 --dtype bfloat16
    import subprocess
    import time

    # Command to start vllm PRM server
    vllm_command = [
        "CUDA_VISIBLE_DEVICES=1",
        "vllm",
        "serve",
        "/research/projects/mllab/public_llms/reward_models/Skywork-Reward-V2-Llama-3.1-8B",
        "--task", "classify",
        "--tensor-parallel-size", "1",
        "--dtype", "bfloat16"
    ]

    # Start the vllm server in the background
    print("Starting vllm (RM) server in the background...")
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

    import argparse

    parser = argparse.ArgumentParser(description="Run ServerRM with optional parameters.")
    parser.add_argument("--prm-model-path", type=str, default="/research/projects/mllab/public_llms/reward_models/Skywork-Reward-V2-Llama-3.1-8B", help="Path to the Reward Model")
    parser.add_argument("--summary-model", type=str, default="gpt-5-mini", help="Model to use for summarization")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000/classify", help="URL for the PRM API endpoint")
    parser.add_argument("--max-parallel-calls", type=int, default=30, help="Max parallel calls for evaluation")
    parser.add_argument("--summary-mode", type=str, choices=["enforce", "optional"], default="enforce", help="Summary mode")
    parser.add_argument("--max-context-length", type=int, default=16384, help="Max context length before triggering summarization")

    args = parser.parse_args()

    server_kwargs = {
        "model": args.prm_model_path,
        "api_url": args.api_url,
        "max_parallel_calls": args.max_parallel_calls,
        "summary_mode": args.summary_mode,
        "summary_model": args.summary_model,
        "max_context_length": args.max_context_length,
    }

    print("Starting ServerRM...")
    server = ServerRM(**server_kwargs)
    server.start()
