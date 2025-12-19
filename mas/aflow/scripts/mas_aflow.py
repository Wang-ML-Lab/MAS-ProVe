from mas_proceval.mas.mas_base import MASBase
from mas_proceval.decorators.decorator_base import llm_parallel_search_decorator


class MASAFlow(MASBase):
    """
    MAS wrapper for AFlow workflows - analogous to MASDebate for debate.
    Wraps workflow execution with trajectory tracking and process evaluation.
    
    Usage (parallel to MASDebate):
        mas_aflow = MASAFlow(workflow, problem, expected_answer, task_type)
        result = await mas_aflow.run()
    """
    
    def __init__(self, workflow, task_type: str = "math"):
        super().__init__()
        self.workflow = workflow
        self.task_type = task_type
        self.problem = ""  # Will be set per call
        
        # Inject decorators into workflow operators (only once)
        self._inject_decorators()
    
    def _inject_decorators(self):
        """
        Inject trajectory-aware decorators into workflow operators.
        Replace operator instances with wrappers that call through MASAFlow methods.
        """
        # Wrap Custom operator
        if hasattr(self.workflow, 'custom'):
            self._original_custom = self.workflow.custom
            
            async def custom_wrapper(*args, **kwargs):
                # Inject kwargs BEFORE calling the decorated method
                kwargs['trajectory'] = self.trajectory.copy()
                kwargs['task_type'] = self.task_type
                if 'question' not in kwargs:
                    kwargs['question'] = self.problem
                return await self._call_custom(*args, **kwargs)
            
            self.workflow.custom = custom_wrapper
        
        # Handle optional operators
        if hasattr(self.workflow, 'programmer'):
            self._original_programmer = self.workflow.programmer
            
            async def programmer_wrapper(*args, **kwargs):
                kwargs['trajectory'] = self.trajectory.copy()
                kwargs['task_type'] = self.task_type
                if 'question' not in kwargs:
                    kwargs['question'] = self.problem
                return await self._call_programmer(*args, **kwargs)
            
            self.workflow.programmer = programmer_wrapper
        
        if hasattr(self.workflow, 'sc_ensemble'):
            self._original_sc_ensemble = self.workflow.sc_ensemble
            
            async def sc_ensemble_wrapper(*args, **kwargs):
                kwargs['trajectory'] = self.trajectory.copy()
                kwargs['task_type'] = self.task_type
                if 'question' not in kwargs:
                    kwargs['question'] = self.problem
                return await self._call_sc_ensemble(*args, **kwargs)
            
            self.workflow.sc_ensemble = sc_ensemble_wrapper
        
        if hasattr(self.workflow, 'answer_generate'):
            self._original_answer_generate = self.workflow.answer_generate
            
            async def answer_generate_wrapper(*args, **kwargs):
                kwargs['trajectory'] = self.trajectory.copy()
                kwargs['task_type'] = self.task_type
                if 'question' not in kwargs:
                    kwargs['question'] = self.problem
                return await self._call_answer_generate(*args, **kwargs)
            
            self.workflow.answer_generate = answer_generate_wrapper
    
    @MASBase.update_trajectory
    @llm_parallel_search_decorator
    async def _call_custom(self, *args, **kwargs):
        """Decorated custom operator call"""
        result = await self._original_custom(*args, **kwargs)
        return result
    
    @MASBase.update_trajectory
    @llm_parallel_search_decorator
    async def _call_programmer(self, *args, **kwargs):
        """Decorated programmer operator call"""
        return await self._original_programmer(*args, **kwargs)
    
    @MASBase.update_trajectory
    @llm_parallel_search_decorator
    async def _call_sc_ensemble(self, *args, **kwargs):
        """Decorated sc_ensemble operator call"""
        return await self._original_sc_ensemble(*args, **kwargs)
    
    @MASBase.update_trajectory
    @llm_parallel_search_decorator
    async def _call_answer_generate(self, *args, **kwargs):
        """Decorated answer_generate operator call"""
        return await self._original_answer_generate(*args, **kwargs)
    
    async def run(self, problem: str):
        """
        Execute the workflow with trajectory tracking for a specific problem.
        Returns same format as normal workflow: (final_answer, cost)
        """
        # Update problem for this execution
        self.problem = problem
        # Reset trajectory for new problem
        self.trajectory = []
        # Execute workflow
        result = await self.workflow(problem)
        return result
