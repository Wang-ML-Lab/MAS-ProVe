from .server_base import BaseServer
from ..utils import read_config_yaml
import asyncio
global_config = read_config_yaml()
import warnings

from litellm import acompletion

class ServerJudge(BaseServer):
    def __init__(self,
                 host=global_config["server"]["host"],
                 port=global_config["server"]["port"],
                 model="gpt-5-mini",
                 max_parallel_calls=50):
        super().__init__(host, port)

        self.openai_api_key = global_config["api_keys"]["openai"]
        self.model_name = model
        self.max_parallel_calls = max_parallel_calls
        self.semaphore = asyncio.Semaphore(max_parallel_calls)

    def format_trajectory_for_eval(self, trajectory):
        # here we leave the autonomacy fo the context generation to the MAS systems.
        context = trajectory["context"] if isinstance(trajectory["context"], str) else str(trajectory["context"])
        current_step = trajectory["current-step"] if isinstance(trajectory["current-step"], str) else str(trajectory["current-step"])
        return "\n".join([context, current_step])
    
    def feedback2rankings(self, feedback, trajectories_for_eval):
        num_candidates = len(trajectories_for_eval)
        print(num_candidates)
        if "<ranking>" in feedback and "</ranking>" in feedback:
            start = feedback.find("<ranking>") + len("<ranking>")
            end = feedback.find("</ranking>")
            ranking_text = feedback[start:end].strip()
        
            try:
                # Parse comma-separated ranking (1-indexed from judge)
                ranks = [int(x.strip()) - 1 for x in ranking_text.split(",")]
                
                # Validate ranking
                if len(ranks) == num_candidates and set(ranks) == set(range(num_candidates)):
                    return ranks
            except (ValueError, IndexError):
                pass
        # Fallback: return sequential ranking
        print(f"Warning: Could not parse ranking, using fallback [0,1,2,...]")
        return list(range(num_candidates))
    
    async def _process_request(self, request):
        assert request["judge-type"] == "judge", "Invalid judge type"
        task_type = request["task-type"]
        partial_trajectories = request["partial_trajectories"]
        question = request.get("question", "")
        
        # format the trajectories for evaluation
        trajectories_for_eval = [
            self.format_trajectory_for_eval(traj) for traj in partial_trajectories
        ]

        messages = build_messages_for_judge(trajectories_for_eval, task_type, question)

        async with self.semaphore:
            response = await acompletion(
                model=self.model_name,
                messages=messages,
                api_key=self.openai_api_key,
            )
        judge_feedback = response.choices[0].message.content
        rankings = self.feedback2rankings(judge_feedback, trajectories_for_eval)

        return {
            "messages-for-proc-eval": messages,
            "judge-feedback": judge_feedback,
            "rankings": rankings,
        }

    def process_request(self, request):
        return asyncio.run(self._process_request(request))


def build_messages_for_judge(trajectories_for_eval, task_type=None, question=""): 
    # (vishal) implement this function with our current prompt template.
    # should be in the format of: 
    # [
    #     {"role": "system", "content": content},
    #     {"role": "user", "content": trajectories_for_eval},
    # ]
    PROMPT_RANKING = """
    <|User Prompt|>
    {question}

    {candidates_text}
    """.strip()

    if "qa" in task_type.lower() or "math" in task_type.lower():
        # Use ranking-based judging
        system_prompt = """Please act as an impartial judge and evaluate the quality of multiple responses provided by AI assistants to the user prompt displayed below. You will be given several candidate responses.

Your task is to rank these responses from best to worst based on their correctness, reasoning quality, and completeness. Consider:
- Correctness of the final answer
- Quality and clarity of reasoning
- Completeness of the solution
- Logical consistency

Be as objective as possible. Avoid any biases such as length or stylistic elements like formatting.

Before providing your ranking, think through the evaluation process and output your thoughts as an explanation.

After providing your explanation, you must output the ranking as a comma-separated list of candidate numbers (e.g., "2,1,3" means candidate 2 is best, candidate 1 is second, candidate 3 is worst). Please enclose your ranking in <ranking> and </ranking> tags.
""".strip()
        candidates_parts = []
        for i, trajectory in enumerate(trajectories_for_eval,1):
            candidates_parts.append(f"<|Candidate {i} Response|>\n{trajectory}\n<|End of Candidate {i}|>")
        
        candidates_text = "\n\n".join(candidates_parts)
        
        prompt_formatted = PROMPT_RANKING.format(
            question=question,
            candidates_text=candidates_text
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_formatted}
        ]
        return messages
    # elif task_type == "swe":
    #     pass
    # elif task_type == "web-search":
    #     pass
    else: 
        warnings.warn(f"Unknown task type: {task_type}, using the default prompt for judge.")
        messages = [
            {"role": "system", "content": "You are an impartial judge who's good at reasoning and deciding the quality of the given complex systems' trajectories. Judge the quality of the given trajectories in a fair and objective manner and give out the rankings of them."},
            {"role": "user", "content": trajectories_for_eval},
        ]
    return messages


if __name__ == "__main__":
    print("Starting ServerJudge...")
    server = ServerJudge(model="gpt-5-mini", max_parallel_calls=100, port=5556)
    print(f"Judge server listening on {server.host}:{server.port}")
    server.start()