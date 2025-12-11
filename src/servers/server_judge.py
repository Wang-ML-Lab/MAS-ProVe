from openai import Client
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
                 max_parallel_calls=10):
        super().__init__(host, port)

        self.openai_api_key = global_config["api_keys"]["openai"]
        self.model_name = model
        self.max_parallel_calls = max_parallel_calls
        self.semaphore = asyncio.Semaphore(max_parallel_calls)

    def format_trajectory_for_eval(self, trajectory):
        # here we leave the autonomacy fo the context generation to the MAS systems.
        return "\n".join([trajectory["context"], trajectory["current-step"]])
    
    def feedback2rankings(self, feedback, trajectories_for_eval):
        # (vishal) implement this function with our current prompt template.
        # here is a simple placeholder: 
        return list(range(len(trajectories_for_eval)))
    
    async def _process_request(self, request):
        assert request["judge-type"] == "judge", "Invalid judge type"
        task_type = request["task-type"]
        partial_trajectories = request["partial_trajectories"]

        # format the trajectories for evaluation
        trajectories_for_eval = [
            self.format_trajectory_for_eval(traj) for traj in partial_trajectories
        ]

        messages = build_messages_for_judge(trajectories_for_eval, task_type)

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


def build_messages_for_judge(trajectories_for_eval, task_type=None): 
    # (vishal) implement this function with our current prompt template.
    # should be in the format of: 
    # [
    #     {"role": "system", "content": content},
    #     {"role": "user", "content": trajectories_for_eval},
    # ]

    if task_type == "math":
        pass 
    elif task_type == "swe":
        pass
    elif task_type == "web-search":
        pass
    else: 
        warnings.warn(f"Unknown task type: {task_type}, using the default prompt for judge.")
        messages = [
            {"role": "system", "content": "You are an impartial judge who's good at reasoning and deciding the quality of the given complex systems' trajectories. Judge the quality of the given trajectories in a fair and objective manner and give out the rankings of them."},
            {"role": "user", "content": trajectories_for_eval},
        ]
    return messages