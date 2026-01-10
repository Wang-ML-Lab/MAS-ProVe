from litellm import acompletion, completion
import requests
from .server_base import BaseServer
from ..utils import read_config_yaml
import asyncio
import warnings
import re
import aiohttp
import argparse
import subprocess
import time

global_config = read_config_yaml()

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


PROMPT_RANKING_SYSTEM = """
Please act as an impartial judge and evaluate the quality of the responses provided by three AI assistants to the user prompt displayed below. You will be given assistant A's answer, assistant B's answer, and assistant C's answer. Your job is to determine which assistant's answer is better.
If assistant A is better, output [A]. If assistant B is better, output [B]. If assistant C is better, output [C].

Here are some rules for evaluation
(1) When evaluating the assistants' answers, identify any mistakes or inaccurate information. Focus on the content each response and select the response that is logically sound and error free.
(2) If both responses contain inaccurate information, select the response that arrives at the correct response
(3) Avoid any biases, such as order of responses, length, or stylistic elements like formatting

Before outputting your final judgment, provide an explanation of your judgment. Your explanation should discuss why your chosen response is better based on the evaluation criteria. The explanation should concretely discuss strengths and weaknesses of both answers.
After outputting your explanation, provide your final judgment. Use the following format:
Explanation: Your explanation here
Verdict: Your final verdict
""".strip()

PROMPT_PAIRWISE = """
[User Question]
{instruction}

[The Start of Assistant A's Answer]
{response_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{response_b}
[The End of Assistant B's Answer]

[The Start of Assistant C's Answer]
{response_c}
[The End of Assistant C's Answer]
""".strip()


class ServerRM(BaseServer):
    def __init__(self,
                 host=None,
                 port=None,
                 model=None,
                 api_url="http://localhost:8000",
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
            {"role": "system", "content": SYSTEM_PROMPT.format(
                max_context_length=self.max_context_length)},
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
        context = candidate["context"] if isinstance(
            candidate["context"], str) else str(candidate["context"])
        current_step = candidate["current-step"] if isinstance(
            candidate["current-step"], str) else str(candidate["current-step"])
        # print(f"[DEBUG] Formatting candidate for eval: context={context[:50]}, current_step={current_step[:50]}")
        return "\n".join([context, current_step])

    def _assemble_conversation_str(self, query, responses):
        if "fare" in self.model.lower():
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.model, trust_remote_code=True)
            # print(f"[DEBUG] Assembling conversation string for model: {self.model}")
            # from the skywork's huggingface repo:
            messages = [
                {"role": "system", "content": PROMPT_RANKING_SYSTEM},
                {"role": "user", "content": PROMPT_PAIRWISE.format(
                    instruction=query,
                    response_a=responses[0],
                    response_b=responses[1],
                    response_c=responses[2],
                )},
            ]
            conversation_str = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )

            return messages, conversation_str
        else:
            raise ValueError(f"Unsupported model: {self.model}")

    def _post_http_request(self, prompt: dict):
        headers = {"User-Agent": "RM-Server"}
        # print(f"[DEBUG] Posting HTTP request to {self.api_url} with prompt: {prompt}")
        response = requests.post(self.api_url, headers=headers, json=prompt)
        # print(f"[DEBUG] HTTP response status: {response.status_code}")
        return response

    def feedback2rankings(self, feedback):
        """
        Parse verdict of the form:
        '# Verdict: [A]' or '# Verdict: [B]', or '# Verdict: [C]'
        Returns a list of rankings, e.g. [0,1,2] meaning A>B>C, [2,1,0] meaning C>B>A, etc.
        """
        # Support modern verdicts that may have `[A]`, `[B]`, `[C]` anywhere in the feedback
        verdict_pattern = r'\[([ABC])\]'
        verdict_match = re.search(verdict_pattern, feedback)
        # print(f"[DEBUG] feedback2rankings(): feedback={feedback!r}, verdict_match={verdict_match}")

        print(verdict_match)

        # Map from letter to ranking index (0=A, 1=B, 2=C)
        letter2idx = {'A': 0, 'B': 1, 'C': 2}

        if verdict_match:
            winner_letter = verdict_match.group(1)
            if winner_letter in letter2idx:
                winner_idx = letter2idx[winner_letter]
                # (Winner is first, others order after)
                ranking = [winner_idx] + \
                    [i for i in range(3) if i != winner_idx]
                return ranking

        # Fallback: print warning and just use sequential
        print(
            "Warning: Could not parse verdict from feedback, using fallback ranking [0,1,2]")
        return [0, 1, 2]

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
            tasks = [self.summarize_context(response)
                     for response in responses]
            summarized_responses = await asyncio.gather(*tasks)
            responses = summarized_responses
        # print(f"[DEBUG] Summarized responses: {responses}")

        messages, conversation_str = self._assemble_conversation_str(
            query, responses)
        print(conversation_str)
        semaphore = await self._get_semaphore()
        async with semaphore:
            try:
                prompt = {
                    "model": self.model,
                    "messages": messages,
                }

                async with aiohttp.ClientSession() as session:
                    async with session.post(self.api_url, json=prompt) as response:
                        response = await response.json()

                judge_feedback = response["choices"][0]["message"]["content"]
                print(judge_feedback)
                rankings = self.feedback2rankings(judge_feedback)
                print(rankings)
            except Exception as e:
                warnings.warn(
                    f"Exception occurred during acompletion reward request: {e}; assigning all rewards -inf"
                )
                judge_feedback = "Error occurred during acompletion reward request"
                print(
                    f"Warning: Could not parse ranking, using fallback [0,1,2,...]")
                rankings = list(range(responses))

        return {
            "messages-for-proc-eval": conversation_str,
            "judge-feedback": judge_feedback,
            "rankings": rankings,
        }

    def process_request(self, request):
        return asyncio.run(self._process_request(request))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run ServerRM with optional parameters.")
    parser.add_argument("--prm-model-path", type=str,
                        default="/research/projects/mllab/public_llms/FARE-20B", help="Path to the FARE model")
    parser.add_argument("--summary-model", type=str,
                        default="gpt-5-mini", help="Model to use for summarization")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000/v1/chat/completions",
                        help="URL for the PRM API endpoint")
    parser.add_argument("--max-parallel-calls", type=int,
                        default=30, help="Max parallel calls for evaluation")
    parser.add_argument("--summary-mode", type=str,
                        choices=["enforce", "optional"], default="optional", help="Summary mode")
    parser.add_argument("--max-context-length", type=int, default=16384,
                        help="Max context length before triggering summarization")

    args = parser.parse_args()

    # To run the PRM server in the shell, execute the following command:
    # CUDA_VISIBLE_DEVICES=0 vllm serve "/research/projects/mllab/public_llms/reward_models/Skywork-Reward-V2-Llama-3.1-8B" --task classify --tensor-parallel-size 1 --dtype bfloat16

    # Command to start vllm PRM server
    vllm_command = [
        "CUDA_VISIBLE_DEVICES=0",
        "vllm",
        "serve",
        args.prm_model_path,
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
