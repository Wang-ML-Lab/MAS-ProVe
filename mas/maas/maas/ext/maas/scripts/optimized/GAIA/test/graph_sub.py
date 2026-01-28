import torch
import maas.ext.maas.scripts.optimized.GAIA.test.template.prompt as prompt_custom
import maas.ext.maas.scripts.optimized.GAIA.test.template.operator as operator
from maas.ext.maas.scripts.optimized.GAIA.test.template.operator_registry import operator_mapping, operator_names
from maas.provider.llm_provider_registry import create_llm_instance
from maas.utils.cost_manager import CostManager
from maas.logs import logger
from maas.tools.web_search_tool import WEB_SEARCH_TOOL, TOOL_FUNCTIONS
from mas_proceval.decorators.decorator_base import llm_parallel_search_decorator

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
        
        # Enable web search tool calling for GAIA dataset
        self.llm.tools = [WEB_SEARCH_TOOL]
        self.llm.tool_functions = TOOL_FUNCTIONS
        # logger.info(f"🌐 GAIA Workflow: Web search tool calling ENABLED")
        # logger.info(f"🔍 LLM instance has tools: {hasattr(self.llm, 'tools')}, value: {getattr(self.llm, 'tools', None) is not None}")
        # logger.info(f"🔍 LLM instance has tool_functions: {hasattr(self.llm, 'tool_functions')}, value: {getattr(self.llm, 'tool_functions', None) is not None}")
        
        self.custom = operator.Generate(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

        self.controller = controller.to(self.device)
        self.operator_embeddings = operator_embeddings.to(self.device)
        self.selection_operator_instances = {
            operator_name: operator_mapping[operator_name](self.llm)
            for operator_name in operator_names
        }
        self.selection_operator_names = operator_names
        
    @llm_parallel_search_decorator
    async def execute_generate(self, operator, input, instruction, **kwargs):
        """Decorated Generate operator - generates 3 candidates, judge picks best"""
        result = await operator(input=input, instruction=instruction)
        return result.get('response', "")
    
    @llm_parallel_search_decorator
    async def execute_self_refine(self, operator, problem, solution, **kwargs):
        """Decorated SelfRefine operator - generates 3 candidates, judge picks best"""
        result = await operator(problem=problem, solution=solution)
        return result.get('response', "")
        
    @llm_parallel_search_decorator
    async def execute_custom(self, input, instruction, **kwargs):
        """Decorated custom operator - generates 3 candidates, judge picks best"""
        result = await self.custom(input=input, instruction=instruction)
        return result.get('response', "")
    
    async def __call__(self, problem: str):
        log_probs_layers, selected_names_layers = self.controller.forward(problem, self.operator_embeddings, self.selection_operator_names)
        
        current_solution = "" 
        solutions = []
        sum_log_prob = 0.0
        
        # Generate initial solution - LLM will autonomously call web_search if needed
        initial_solution = await self.execute_custom(
            input=problem,
            instruction=prompt_custom.GENERATE_SOLUTION_PROMPT,
            task_type="qa",
            question=problem,
            trajectory=trajectory.copy()
        )
        current_solution = initial_solution
        solutions.append(current_solution)
        trajectory.append(f"Initial Solution: {current_solution}...")

        for layer_idx, selected_names in enumerate(selected_names_layers):
            for op_name in selected_names:
                selected_operator = self.selection_operator_instances[op_name]

                if op_name in ["Generate", "GenerateCoT"]:
                    # Use decorated method with process evaluation
                    new_solution = await self.execute_generate(
                        operator=selected_operator,
                        input=problem,
                        instruction=prompt_custom.DETAILED_SOLUTION_PROMPT,
                        task_type="qa",
                        question=problem,
                        trajectory=trajectory.copy()
                    )
                    solutions.append(new_solution)
                    trajectory.append(f"{op_name}: {new_solution}...")
                    
                elif op_name == "SelfRefine":
                    # Use decorated method with process evaluation
                    new_solution = await self.execute_self_refine(
                        operator=selected_operator,
                        problem=problem,
                        solution=current_solution,
                        task_type="qa",
                        question=problem,
                        trajectory=trajectory.copy()
                    )
                    solutions.append(new_solution)
                    trajectory.append(f"SelfRefine: {new_solution}...")                
                elif op_name == "ScEnsemble":
                    result = await selected_operator(problem=problem, solutions=solutions)
                    solutions = []
                    new_solution = result.get('response', "")
                    solutions.append(new_solution)
                    trajectory.append(f"ScEnsemble: {new_solution}...")
                    
                elif op_name == "MultiGenerateCoT":
                    result = await selected_operator(input=problem, instruction=prompt_custom.GENERATE_SOLUTION_PROMPT)
                    if isinstance(result, dict) and 'response' in result:
                        for res in result['response']:
                            new_solution = res.get('response', "")
                            solutions.append(new_solution)
                        trajectory.append(f"MultiGenerateCoT: {len(result['response'])} solutions")
                    else:
                        logger.error(f"Expected dict with 'responses' from MultiGenerateCoT, got {type(result)}")
                        new_solution = current_solution
                else:
                    new_solution = current_solution

                current_solution = new_solution

            sum_log_prob += log_probs_layers[layer_idx]

        if len(solutions) > 1:
            final_solution = await self.sc_ensemble(solutions=solutions, problem=problem)
            final_solution = final_solution['response']
        else:
            final_solution = current_solution

        return final_solution, self.llm.cost_manager.total_cost, sum_log_prob
