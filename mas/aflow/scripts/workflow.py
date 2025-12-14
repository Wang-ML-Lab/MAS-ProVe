# -*- coding: utf-8 -*-
# @Date    : 6/27/2024 22:07 PM
# @Author  : didi
# @Desc    : Basic Graph Class


from scripts.evaluator import DatasetType
from scripts.async_llm import create_llm_instance
from typing import List, Dict, Callable, Optional

class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self._tools: Optional[List[dict]] = None
        self._tool_functions: Optional[Dict[str, Callable]] = None
    
    @property
    def tools(self) -> Optional[List[dict]]:
        """Get the tools configured for this workflow's LLM"""
        return self._tools
    
    @tools.setter
    def tools(self, value: List[dict]):
        """Set tools on this workflow's LLM"""
        # print(f"[WORKFLOW] Setting tools on workflow LLM")
        self._tools = value
        self.llm.tools = value
    
    @property
    def tool_functions(self) -> Optional[Dict[str, Callable]]:
        """Get the tool functions configured for this workflow's LLM"""
        return self._tool_functions
    
    @tool_functions.setter
    def tool_functions(self, value: Dict[str, Callable]):
        """Set tool functions on this workflow's LLM"""
        # print(f"[WORKFLOW] Setting tool functions on workflow LLM: {list(value.keys())}")
        self._tool_functions = value
        self.llm.tool_functions = value

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        raise NotImplementedError("This method should be implemented by the subclass")
