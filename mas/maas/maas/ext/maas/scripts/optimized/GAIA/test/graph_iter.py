import torch
import maas.ext.maas.scripts.optimized.GAIA.test.template.prompt as prompt_custom
import maas.ext.maas.scripts.optimized.GAIA.test.template.operator as operator
from maas.ext.maas.scripts.optimized.GAIA.test.template.operator_registry import operator_mapping, operator_names
from maas.provider.llm_provider_registry import create_llm_instance
from maas.utils.cost_manager import CostManager
from maas.logs import logger
from maas.tools.search_engine_ddg import DDGAPIWrapper
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
        self.custom = operator.Generate(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        self.search_engine = DDGAPIWrapper()

        self.controller = controller.to(self.device)
        self.operator_embeddings = operator_embeddings.to(self.device)
        self.selection_operator_instances = {
            operator_name: operator_mapping[operator_name](self.llm)
            for operator_name in operator_names
        }
        self.selection_operator_names = operator_names
    
    async def __call__(self, problem: str):
        """
        Wrapper that triggers the parallel search.
        The decorator runs the logic 3 times in parallel and the Judge picks the best.
        """
        # Call the decorated function with required Judge parameters
        best_result = await self._run_logic_async(
            problem=problem,
            question=problem,      # REQUIRED: User prompt for the Judge
            task_type="qa"    # REQUIRED: Triggers the Math/QA prompt template
        )
        
        # Unpack the best result
        if isinstance(best_result, dict):
            return best_result["response"], best_result["cost"], best_result["log_prob"]
        else:
            return str(best_result), 0, 0.0
     
    @llm_parallel_search_decorator
    async def _run_logic_async(self, problem: str, **kwargs):    
        log_probs_layers, selected_names_layers = self.controller.forward(problem, self.operator_embeddings, self.selection_operator_names)
        
        current_solution = "" 
        solutions = []
        sum_log_prob = 0.0
        
        # Perform initial web search to gather context
        search_query = problem  # Use first 200 chars as query
        search_results = await self.search_engine.run(query=search_query, max_results=5, as_string=True)
        
        # Generate initial solution with search context
        initial_solution = await self.custom(
            input=f"Problem: {problem}\n\nSearch Results: {search_results}", 
            instruction=prompt_custom.GENERATE_SOLUTION_PROMPT
        )
        current_solution = initial_solution['response']
        solutions.append(current_solution)

        for layer_idx, selected_names in enumerate(selected_names_layers):
            for op_name in selected_names:
                selected_operator = self.selection_operator_instances[op_name]

                if op_name in ["Generate", "GenerateCoT"]:
                    # Perform search to enrich context
                    search_results = await self.search_engine.run(query=problem, max_results=3, as_string=True)
                    result = await selected_operator(
                        input=f"Problem: {problem}\n\nSearch Context: {search_results}", 
                        instruction=prompt_custom.DETAILED_SOLUTION_PROMPT
                    )
                    new_solution = result.get('response', "")
                    solutions.append(new_solution)
                elif op_name == "SelfRefine":
                    result = await selected_operator(problem=problem, solution=current_solution)
                    new_solution = result.get('response', "")
                    solutions.append(new_solution)
                elif op_name == "ScEnsemble":
                    result = await selected_operator(problem=problem, solutions=solutions)
                    solutions = []
                    new_solution = result.get('response', "")
                    solutions.append(new_solution)
                elif op_name == "MultiGenerateCoT":
                    result = await selected_operator(input=problem, instruction=prompt_custom.GENERATE_SOLUTION_PROMPT)
                    if isinstance(result, dict) and 'response' in result:
                        for res in result['response']:
                            new_solution = res.get('response', "")
                            solutions.append(new_solution)
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

        return {
            "response": final_solution,      # The reasoning trace (Primary for Judge)
            "cost": self.llm.cost_manager.total_cost,          # Total cost incurred (Primary for Judge)
            "log_prob": sum_log_prob         # Cumulative log probability (Primary for Judge)
            }
