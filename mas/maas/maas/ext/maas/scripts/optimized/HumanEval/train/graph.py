import torch
import maas.ext.maas.scripts.optimized.HumanEval.train.template.prompt as prompt_custom
import maas.ext.maas.scripts.optimized.HumanEval.train.template.operator as operator
from maas.ext.maas.scripts.optimized.HumanEval.train.template.operator_registry import operator_mapping, operator_names
from maas.provider.llm_provider_registry import create_llm_instance
from maas.utils.cost_manager import CostManager
from maas.logs import logger

class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset,
        controller: torch.nn.Module,
        operator_embeddings,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm.cost_manager = CostManager()
        self.test_operator = operator.Test(self.llm)
        self.custom_code_generate = operator.CustomCodeGenerate(self.llm)
        
        self.controller = controller.to(self.device)
        self.operator_embeddings = operator_embeddings.to(self.device)
        self.selection_operator_instances = {
            operator_name: operator_mapping[operator_name](self.llm)
            for operator_name in operator_names
        }
        self.selection_operator_names = operator_names
        
        
    async def __call__(self, problem: str, entry_point: str, log_path: str):
        log_probs_layers, selected_names_layers = self.controller.forward(problem, self.operator_embeddings, self.selection_operator_names)
        
        current_solution = "" 
        solutions = []
        sum_log_prob = 0.0

        for layer_idx, selected_names in enumerate(selected_names_layers):
            for op_name in selected_names:
                selected_operator = self.selection_operator_instances[op_name]

                if op_name in ["Generate", "GenerateCoT"]:
                    result = await selected_operator(problem=problem, entry_point=entry_point, instruction="")
                    new_solution = result.get('response', "")
                    solutions.append(new_solution)
                elif op_name == "SelfRefine":
                    result = await selected_operator(problem=problem, solution=current_solution)
                    new_solution = result.get('response', "")
                    solutions.append(new_solution)
                elif op_name == "Test":
                    result = await selected_operator(problem=problem, solution=current_solution, entry_point=entry_point)
                    new_solution = result.get('solution', "")
                    solutions.append(new_solution)
                elif op_name == "ScEnsemble":
                    result = await selected_operator(problem=problem, solutions=solutions)
                    solutions = []
                    new_solution = result.get('response', "")
                    solutions.append(new_solution)          
                elif op_name == "MultiGenerateCoT":
                    result = await selected_operator(problem=problem, entry_point=entry_point, instruction="")
                    if isinstance(result, dict) and 'response' in result:
                        for res in result['response']:
                            new_solution = res.get('response', "")
                            solutions.append(new_solution)
                    else:
                        logger.error(f"Expected dict with 'response' from MultiGenerateCoT, got {type(result)}")
                        new_solution = current_solution
                else:
                    new_solution = current_solution

                current_solution = new_solution

            sum_log_prob += log_probs_layers[layer_idx]

        test_result = await self.test_operator(problem=problem, solution=current_solution, entry_point=entry_point)

        if test_result['result']:
            return test_result['solution'], self.llm.cost_manager.total_cost, sum_log_prob
        else:
            new_solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=prompt_custom.IMPROVE_CODE_PROMPT)
            return new_solution['response'], self.llm.cost_manager.total_cost, sum_log_prob
