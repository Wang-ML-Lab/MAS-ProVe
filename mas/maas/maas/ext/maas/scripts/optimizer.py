import asyncio
import time
import torch
import os
import numpy as np
from typing import List, Literal

from pydantic import BaseModel, Field
from maas.ext.maas.scripts.evaluator import DatasetType
from maas.ext.maas.scripts.optimizer_utils.data_utils import DataUtils               
from maas.ext.maas.scripts.optimizer_utils.experience_utils import ExperienceUtils
from maas.ext.maas.scripts.optimizer_utils.evaluation_utils import EvaluationUtils
from maas.ext.maas.scripts.optimizer_utils.graph_utils import GraphUtils           
from maas.logs import logger
from maas.ext.maas.models.utils import get_sentence_embedding
from maas.ext.maas.models.controller import MultiLayerController

QuestionType = Literal["math", "code", "qa"]
OptimizerType = Literal["Graph", "Test"]

class GraphOptimize(BaseModel):
    modification: str = Field(default="", description="modification")
    graph: str = Field(default="", description="graph")
    prompt: str = Field(default="", description="prompt")


class Optimizer:
    def __init__(
        self,
        dataset: DatasetType,
        question_type: QuestionType,
        opt_llm_config,
        exec_llm_config,
        operators: List,
        sample: int,
        optimized_path: str = None,
        round: int = 1,
        batch_size: int = 4,
        lr: float = 0.01,
        is_textgrad: bool = False,
    ) -> None:
        self.optimize_llm_config = opt_llm_config
        self.execute_llm_config = exec_llm_config
        self.dataset = dataset
        self.type = question_type
        self.graph = None
        self.operators = operators
        self.root_path = f"{optimized_path}/{self.dataset}"
        self.sample = sample
        self.top_scores = []
        self.round = round
        self.batch_size = batch_size
        self.lr = lr
        self.is_textgrad = is_textgrad
        self.graph_utils = GraphUtils(self.root_path)
        self.data_utils = DataUtils(self.root_path)
        self.experience_utils = ExperienceUtils(self.root_path)
        self.evaluation_utils = EvaluationUtils(self.root_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.controller = MultiLayerController(device=self.device).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.controller.parameters(), lr=self.lr)          

    def optimize(self, mode: OptimizerType = "Graph"):
        if mode == "Test":
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            score = loop.run_until_complete(self.test())
            return None

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        retry_count = 0
        max_retries = 1
        round = 1

        while retry_count < max_retries:
            try:
                score = loop.run_until_complete(self._optimize_graph_maas()) 
                break
            except Exception as e:
                retry_count += 1
                logger.info(f"Error occurred: {e}. Retrying... (Attempt {retry_count}/{max_retries})")
                if retry_count == max_retries:
                    logger.info("Max retries reached. Moving to next round.")
                    score = None

                wait_time = 5 * retry_count
                time.sleep(wait_time)

            if retry_count < max_retries: 
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

        logger.info(f"Score for round {round}: {score}")
        round += 1
        
        time.sleep(5)

    async def _optimize_graph_maas(self):
        graph_path = f"{self.root_path}/train"
        data = self.data_utils.load_results(graph_path)

        operator_descriptions = self.graph_utils.load_operators_description_maas(self.operators) 
        precomputed_operator_embeddings = torch.stack([get_sentence_embedding(op_desc) for op_desc in operator_descriptions]).to(self.device)
        directory = self.graph_utils.create_round_directory(graph_path, self.round)
        logger.info(directory)

        self.graph = self.graph_utils.load_graph_maas(graph_path)

        params = {
            "operator_embeddings": precomputed_operator_embeddings,
            "controller": self.controller,
            "execute_llm_config": self.execute_llm_config, 
            "dataset": self.dataset, 
            "optimizer": self.optimizer,
            "sample": self.sample,
            "is_textgrad": self.is_textgrad
        }

        avg_score = await self.evaluation_utils.evaluate_graph_maas(self, directory, data, initial=False, params=params)

        return avg_score

    async def test(self):
        data = []
        graph_path = f"{self.root_path}/test"
        
        json_file_path = self.data_utils.get_results_file_path(graph_path)
        data = self.data_utils.load_results(graph_path)

        operator_descriptions = self.graph_utils.load_operators_description_maas(self.operators) 
        precomputed_operator_embeddings = torch.stack([get_sentence_embedding(op_desc) for op_desc in operator_descriptions]).to(self.device)

        self.graph = self.graph_utils.load_graph_maas(graph_path)
        directory = self.graph_utils.create_round_directory(graph_path, self.round)

        pth_path = f"{self.root_path}/train"
        pth_directory = self.graph_utils.create_round_directory(pth_path, self.round)
        controller_path = os.path.join(pth_directory,  f"{self.dataset}_controller_sample{self.sample}.pth")
        logger.info(controller_path)

        if os.path.exists(controller_path):
            checkpoint = torch.load(controller_path, map_location=self.device)
            self.controller.load_state_dict(checkpoint)
            self.controller.eval()
        else:
            raise FileNotFoundError(f"Controller model file not found at {controller_path}")         

        params = {
            "operator_embeddings": precomputed_operator_embeddings,
            "controller": self.controller,
            "execute_llm_config": self.execute_llm_config,  
            "dataset": self.dataset,                        
            "optimizer": self.optimizer,
            "sample": self.sample,
            "is_textgrad": False
        }

        score = await self.evaluation_utils.evaluate_graph_test_maas(self, directory, is_test=True, params=params)

        new_data = self.data_utils.create_result_data(self.round, score,0.0,0.0,0)
        data.append(new_data)

        self.data_utils.save_results(json_file_path, data)

        return score
