import json
import os
import re
import time
import traceback
from typing import List

from maas.logs import logger


class GraphUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path

    def create_round_directory(self, graph_path: str, round_number: int) -> str:
        directory = os.path.join(graph_path, f"round_{round_number}")
        os.makedirs(directory, exist_ok=True)
        return directory

    def load_graph_maas(self, workflows_path: str):
        # print(workflows_path)
        workflows_path = workflows_path.replace("\\", ".").replace("/", ".")
        graph_module_name = f"{workflows_path}.graph"
        # print(graph_module_name)

        try:
            graph_module = __import__(graph_module_name, fromlist=[""])
            graph_class = getattr(graph_module, "Workflow")
            return graph_class
        except ImportError as e:
            logger.info(f"Error loading graph: {e}")
            raise

    def read_graph_files(self, round_number: int, workflows_path: str):
        prompt_file_path = os.path.join(workflows_path, f"round_{round_number}", "prompt.py")
        graph_file_path = os.path.join(workflows_path, f"round_{round_number}", "graph.py")

        try:
            with open(prompt_file_path, "r", encoding="utf-8") as file:
                prompt_content = file.read()
            with open(graph_file_path, "r", encoding="utf-8") as file:
                graph_content = file.read()
        except FileNotFoundError as e:
            logger.info(f"Error: File not found for round {round_number}: {e}")
            raise
        except Exception as e:
            logger.info(f"Error loading prompt for round {round_number}: {e}")
            raise
        return prompt_content, graph_content

    def extract_solve_graph(self, graph_load: str) -> List[str]:
        pattern = r"class Workflow:.+"
        return re.findall(pattern, graph_load, re.DOTALL)
    
    def load_operators_description_maas(self, operators: List[str]) -> List[str]:
        path = f"{self.root_path}/train/template/operator.json"
        operators_description = []
        for id, operator in enumerate(operators, start=0):
            operator_description = self._load_operator_description(id, operator, path)
            operators_description.append(operator_description)
        return operators_description
    
    def _load_operator_description(self, id: int, operator_name: str, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            operator_data = json.load(f)
            matched_data = operator_data.get(operator_name, {})
            desc = matched_data.get("description", "No description available")
            interface = matched_data.get("interface", "No interface specified")
            return f"{id}. {operator_name}: {desc}, with interface {interface}."

    async def get_graph_optimize_response(self, graph_optimize_node):
        max_retries = 5
        retries = 0

        while retries < max_retries:
            try:
                response = graph_optimize_node.instruct_content.model_dump() 
                return response
            except Exception as e:
                retries += 1
                logger.info(f"Error generating prediction: {e}. Retrying... ({retries}/{max_retries})")
                if retries == max_retries:
                    logger.info("Maximum retries reached. Skipping this sample.")
                    break
                traceback.print_exc()
                time.sleep(5)
        return None


