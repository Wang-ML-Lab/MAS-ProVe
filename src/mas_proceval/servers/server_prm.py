import requests
from .server_base import BaseServer
from ..utils import read_config_yaml
import asyncio
import warnings
global_config = read_config_yaml()
from litellm import acompletion


SYSTEM_PROMPT = r"""
You are a summarization assistant for multi-agent system processes.

Your task is to summarize the given context into a concise and informative summary.
The total length of the summary should be fewer than 4096 tokens.

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

class ServerPRM(BaseServer):
    def __init__(self,
                 host=None,
                 port=None,
                 model=None,
                 api_url="http://localhost:8000/pooling",
                 max_parallel_calls=50,
                 summary_mode="enforce",
                 summary_model="gpt-5-mini",
                 max_context_length=4096):
        config = global_config.copy()
        host = host or config["server"].get("host", "localhost")
        port = port or config["server"].get("port", 8001)
        super().__init__(host, port)
        self.model = model or config["prm"]["model_path"]
        self.api_url = api_url or config["prm"]["api_url"]
        self.max_parallel_calls = max_parallel_calls

        # summarization should be performed when the context is too long for the PRM model.
        assert summary_mode in ["enforce", "optional"], "Invalid summary mode"
        self.summary_mode = summary_mode
        self.max_context_length = max_context_length
        self.openai_api_key = global_config["api_keys"]["openai"]
        self.summary_model_name = summary_model
    
    async def summarize_context(self, context: str, semaphore) -> str:
        # print("[DEBUG] summarize_context called")
        # print(f"[DEBUG] Context to summarize (first 200 chars): {context[:200]}")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Please summarize the given context into a concise and informative summary. The total length of the summary should be fewer than 4096 tokens.\n\nContext: {context}"},
        ]
        async with semaphore:
            try:
                response = await acompletion(
                    model=self.summary_model_name,
                    messages=messages,
                    api_key=self.openai_api_key,
                    reasoning_effort="minimal",
                    verbosity="low",
                )
                # print("[DEBUG] Summarization response received.")
            except Exception as e:
                # print(f"[ERROR] Exception in summarize_context: {e}")
                raise
        return response.choices[0].message.content

    def format_candidate_for_eval(self, candidate):
        # For PRM, the candidate should be just the 'response'
        context = candidate["context"] if isinstance(candidate["context"], str) else str(candidate["context"])
        current_step = candidate["current-step"] if isinstance(candidate["current-step"], str) else str(candidate["current-step"])
        return "\n".join([context, current_step])
    
    
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
        # print("[DEBUG] _process_request called")
        semaphore = asyncio.Semaphore(self.max_parallel_calls)
        # print(f"[DEBUG] Request: {request}")
        assert request.get(
            "judge-type", None) == "prm", "Invalid judge type for PRM"
        task_type = request.get("task-type", None)
        candidates = request["partial_trajectories"]
        query = request.get("question", "")

        # Prepare candidate responses
        responses = [self.format_candidate_for_eval(candidate)
                     for candidate in candidates]
        # print(f"[DEBUG] Prepared {len(responses)} candidate responses.")

        # summarize the context;
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model, trust_remote_code=True)
        if self.summary_mode == "enforce" or \
            any(len(tokenizer.encode(response)) > self.max_context_length for response in responses):
            # print("[DEBUG] Summarizing context for responses...")
            tasks = [self.summarize_context(response, semaphore) for response in responses]
            summarized_responses = await asyncio.gather(*tasks)
            responses = summarized_responses
        # print(f"[DEBUG] Summarized responses: {responses}")

        conversation_strs = [self._assemble_conversation_str(
            query, response) for response in responses]
        # print(f"[DEBUG] Built {len(conversation_strs)} conversation strings.")
        
        prompts = [{"model": self.model, "input": conversation_str}
                   for conversation_str in conversation_strs]
        # print(f"[DEBUG] Built {len(prompts)} prompts for backend.")

        rewards = []
        async with semaphore:
            # Since requests is not async, run each prompt in thread pool
            import aiohttp

            async with aiohttp.ClientSession(headers={"User-Agent": "PRM-Server"}) as session:
                for idx, prompt in enumerate(prompts):
                    # print(f"[DEBUG] Sending prompt {idx+1}/{len(prompts)} to backend API...")
                    try:
                        async with session.post(self.api_url, json=prompt) as response:
                            # print(f"[DEBUG] Received response with status {response.status}")
                            if response.status != 200:
                                text = await response.text()
                                # print(f"[ERROR] Backend returned status {response.status}: {text}")
                                warnings.warn(
                                    f"Request failed with status code {response.status}: {text}; assigning a reward of 0.0"
                                )
                                reward = 0.0
                            else:
                                try:
                                    resp_json = await response.json()
                                    # print(f"[DEBUG] Backend JSON: {resp_json}")
                                    # Assuming response.json()["data"][0]["data"][0][1] gives the reward for this prompt
                                    reward = resp_json["data"][0]["data"][0][1]
                                except Exception as e:
                                    text = await response.text()
                                    # print(f"[ERROR] Exception parsing backend JSON: {e}\nRaw text: {text}")
                                    raise ValueError(
                                        f"Malformed response from PRM API: {text}"
                                    ) from e
                            rewards.append(reward)
                    except Exception as e:
                        # print(f"[ERROR] Exception during backend call: {e}")
                        raise

        print(f"[DEBUG] Rewards: {rewards}")
        assert len(rewards) == len(
            responses), "Rewards length mismatch with candidates"

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

    import argparse

    parser = argparse.ArgumentParser(description="Run ServerPRM with optional parameters.")
    parser.add_argument("--prm-model-path", type=str, default="/research/projects/mllab/public_llms/reward_models/qwen_rms/Qwen2.5-Math-PRM-7B", help="Path to the PRM model")
    parser.add_argument("--summary-model", type=str, default="gpt-5-mini", help="Model to use for summarization")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000/pooling", help="URL for the PRM API endpoint")
    parser.add_argument("--max-parallel-calls", type=int, default=20, help="Max parallel calls for evaluation")
    parser.add_argument("--summary-mode", type=str, choices=["enforce", "optional"], default="enforce", help="Summary mode")
    parser.add_argument("--max-context-length", type=int, default=4096, help="Max context length before triggering summarization")

    args = parser.parse_args()

    server_kwargs = {
        "model": args.prm_model_path,
        "api_url": args.api_url,
        "max_parallel_calls": args.max_parallel_calls,
        "summary_mode": args.summary_mode,
        "summary_model": args.summary_model,
        "max_context_length": args.max_context_length,
    }

    print("Starting ServerPRM...")
    server = ServerPRM(**server_kwargs)
    server.start()