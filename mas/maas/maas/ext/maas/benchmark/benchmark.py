import asyncio
import json
import os
import torch
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Tuple
from pydantic import BaseModel, Field
from maas.actions.action_node import ActionNode
import aiofiles
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from maas.configs.models_config import ModelsConfig
from maas.provider.llm_provider_registry import create_llm_instance
from maas.logs import logger
from maas.utils.common import write_json_file
from maas.ext.maas.scripts.utils import extract_random_prompt, update_prompt_in_file
from maas.ext.maas.scripts.textgrad.textual_gradient import TEXT_GRAD_PROMPT

class TextGrad(BaseModel):
    prompt: str = Field(default="", description="prompt")

class BaseBenchmark(ABC):
    def __init__(
        self,
        name: str,
        file_path: str,
        log_path: str,
        batch_size: int,
        controller: torch.nn.Module,
        operator_embeddings,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.name = name
        self.file_path = file_path
        self.log_path = log_path
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.controller = controller.to(self.device)
        self.operator_embeddings = operator_embeddings.to(self.device)
        self.optimizer = optimizer

    PASS = "PASS"
    FAIL = "FAIL"

    async def load_data(self, specific_indices: List[int] = None) -> List[dict]:
        data = []
        async with aiofiles.open(self.file_path, mode="r", encoding="utf-8") as file:
            async for line in file:
                data.append(json.loads(line))
        if specific_indices is not None:
            filtered_data = [data[i] for i in specific_indices if i < len(data)]
            return filtered_data
        return data

    def save_results_to_csv(self, results: List[Tuple[Any, ...]], columns: List[str]):
        df = pd.DataFrame(results, columns=columns)
        avg_score = df["score"].mean()
        if "cost" in df.columns:
            df = df.drop(columns=["cost"])
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{avg_score:.5f}_{current_time}.csv"
        output_file = os.path.join(self.log_path, filename)
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        return avg_score

    def log_mismatch(
        self,
        problem: str,
        expected_output: Any,
        prediction: str,
        extracted_output: Any,
        extract_answer_code: str = "None",
    ):
        log_data = {
            "question": problem,
            "right_answer": expected_output,
            "model_output": prediction,
            "extracted_output": extracted_output,
            "extract_answer_code": extract_answer_code,
        }
        log_file = Path(self.log_path) / "log.json"
        if log_file.exists():
            with log_file.open("r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []
        data.append(log_data)
        write_json_file(log_file, data, encoding="utf-8", indent=4)

    @abstractmethod
    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[Any, ...]:
        pass

    @abstractmethod
    def calculate_score(self, expected_output: Any, prediction: Any) -> Tuple[float, Any]:
        pass

    @abstractmethod
    def get_result_columns(self) -> List[str]:
        pass

    async def evaluate_all_problems(self, data: List[dict], graph: Callable, max_concurrent_tasks: int = 10, repetitions: int = 4, is_textgrad: bool = False):
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        results = []
        previous_cost = 0.0
        textgrad = False           
        prev_rep_score = None   

        async def sem_evaluate(problem):
            async with semaphore:
                try:
                    return await self.evaluate_problem(problem, graph)
                except Exception as e:
                    logger.error(f"Error evaluating problem: {e}")
                    return ("", "", "", 0.0, 0.0, 0.0)
        
        for rep in range(1, repetitions + 1):
            logger.info(f"Starting training repetition {rep}/{repetitions}")
            rep_scores = []

            if textgrad and is_textgrad:
                prompt_name, prompt_content = extract_random_prompt(self.log_path)
                textgrad_prompt = TEXT_GRAD_PROMPT.format(dataset = self.name, prompt_name = prompt_name, prompt_content = prompt_content)
                textgrad_llm_config = ModelsConfig.default().get("gpt-4o-mini")
                textgrad_llm = create_llm_instance(textgrad_llm_config)
                textgrad_node = await ActionNode.from_pydantic(TextGrad).fill(context=textgrad_prompt, mode="xml_fill", llm=textgrad_llm)
                response = textgrad_node.instruct_content.model_dump()
                update_prompt_in_file(prompt_name, response["prompt"])
                is_textgrad = False

            for batch_start in range(0, len(data), self.batch_size):
                batch = data[batch_start:batch_start + self.batch_size]
                tasks = [sem_evaluate(problem) for problem in batch]
                batch_results = await tqdm_asyncio.gather(
                    *tasks, 
                    desc=f"Repetition {rep}: Executing batch {batch_start // self.batch_size + 1}", 
                    total=len(batch)
                )
                results.extend(batch_results)

                logprobs = []
                scores = []
                costs = []
                for r in batch_results:
                    logprob = r[5]
                    cost = r[4]
                    score = r[3]
                    logprobs.append(logprob)
                    scores.append(score)
                    costs.append(cost - previous_cost)
                    previous_cost = cost
                    rep_scores.append(score)

                if len(logprobs) > 0:
                    logprobs = torch.stack(logprobs).to(self.device)
                    scores_tensor = torch.tensor(scores, dtype=torch.float32, device=self.device)
                    costs_tensor = torch.tensor(costs, dtype=torch.float32, device=self.device)
                    utilities = scores_tensor - 3 * costs_tensor
                    loss = -(logprobs * utilities).mean()
                    if loss.requires_grad:
                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        logger.info(f"Repetition {rep}: Batch {batch_start // self.batch_size + 1} Loss: {loss.item()}")
                    else:
                        logger.info(f"Repetition {rep}: Batch {batch_start // self.batch_size + 1} Loss does not require grad and was skipped.")
                else:
                    logger.info(f"Repetition {rep}: Batch {batch_start // self.batch_size + 1} skipped due to invalid logprobs.")

            if rep_scores:
                current_rep_score = sum(rep_scores) / len(rep_scores)
            else:
                current_rep_score = 0.0

            if not textgrad:
                if prev_rep_score is not None and current_rep_score < prev_rep_score:
                    textgrad = True
                prev_rep_score = current_rep_score

        return results
    
    async def evaluate_all_problems_test(self, data: List[dict], graph: Callable, max_concurrent_tasks: int = 20):
        semaphore = asyncio.Semaphore(max_concurrent_tasks)

        async def sem_evaluate(problem):
            async with semaphore:
                return await self.evaluate_problem(problem, graph)

        tasks = [sem_evaluate(problem) for problem in data]
        return await tqdm_asyncio.gather(*tasks, desc=f"Evaluating {self.name} problems", total=len(data))
    
    async def run_evaluation(self, graph: Callable, va_list: List[int], is_test: bool, sample: int, is_textgrad: bool = False, max_concurrent_tasks: int = 10):
        data = await self.load_data(va_list)

        if is_test == True:
            results = await self.evaluate_all_problems_test(data, graph, max_concurrent_tasks)
            columns = self.get_result_columns()
            average_score = self.save_results_to_csv(results, columns)
            logger.info(f"Average score on {self.name} dataset: {average_score:.5f}")
                
            return average_score
        
        results = await self.evaluate_all_problems(data, graph, max_concurrent_tasks, sample, is_textgrad)

        columns = self.get_result_columns()
        average_score = self.save_results_to_csv(results, columns)
        logger.info(f"Average score on {self.name} dataset: {average_score:.5f}")
        
        try:
            os.makedirs(self.log_path, exist_ok=True)
            controller_path = os.path.join(self.log_path, f"{self.name}_controller_sample{sample}.pth")
            torch.save(self.controller.state_dict(), controller_path)
            logger.info(f"Saved controller parameters to {controller_path}")     
            logger.info("Successfully Finish Training")       
        except Exception as e:
            logger.error(f"Failed to save controller parameters: {e}")       

        return average_score
