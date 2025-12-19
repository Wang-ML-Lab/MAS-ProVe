# -*- coding: utf-8 -*-
# @Date    : 8/23/2024 10:00 AM
# @Author  : all
# @Desc    : Evaluation for different datasets

from typing import Dict, Literal, Tuple

from benchmarks.benchmark import BaseBenchmark
from benchmarks.drop import DROPBenchmark
from benchmarks.gsm8k import GSM8KBenchmark
from benchmarks.hotpotqa import HotpotQABenchmark
from benchmarks.humaneval import HumanEvalBenchmark
from benchmarks.math import MATHBenchmark
from benchmarks.mbpp import MBPPBenchmark
from benchmarks.livecodebench import LiveCodeBench
from benchmarks.aime24 import AIME24Benchmark
from benchmarks.aime25 import AIME25Benchmark
from benchmarks.gaia import GAIABenchmark
from scripts.mas_aflow import MASAFlow

# If you want to customize tasks, add task types here and provide evaluation functions, just like the ones given above
DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP", "LiveCodeBench", "AIME24", "AIME25", "GAIA"]


class Evaluator:
    """
    Complete the evaluation for different datasets here
    """

    def __init__(self, eval_path: str):
        self.eval_path = eval_path
        self.dataset_configs: Dict[DatasetType, BaseBenchmark] = {
            "GSM8K": GSM8KBenchmark,
            "MATH": MATHBenchmark,
            "HumanEval": HumanEvalBenchmark,
            "HotpotQA": HotpotQABenchmark,
            "MBPP": MBPPBenchmark,
            "DROP": DROPBenchmark,
            "LiveCodeBench": LiveCodeBench,
            "AIME24": AIME24Benchmark,
            "AIME25": AIME25Benchmark,
            "GAIA": GAIABenchmark,
        }
        
        # Map datasets to their task types
        self.task_types: Dict[DatasetType, str] = {
            "GSM8K": "math",
            "MATH": "math",
            "AIME24": "math",
            "AIME25": "math",
            "HumanEval": "code",
            "MBPP": "code",
            "LiveCodeBench": "code",
            "HotpotQA": "qa",
            "DROP": "qa",
            "GAIA": "qa",
        }

    async def graph_evaluate(
        self, dataset: DatasetType, graph, params: dict, path: str, is_test: bool = False
    ) -> Tuple[float, float, float]:
        if dataset not in self.dataset_configs:
            raise ValueError(f"Unsupported dataset: {dataset}")

        data_path = self._get_data_path(dataset, is_test)
        benchmark_class = self.dataset_configs[dataset]
        benchmark = benchmark_class(name=dataset, file_path=data_path, log_path=path)

        # Use params to configure the graph and benchmark
        configured_graph = await self._configure_graph(dataset, graph, params)
        if is_test:
            va_list = None # For test data, generally use None to test all
        else:
            va_list = [2] # Use None to test all Validation data, or set va_list (e.g., [1, 2, 3]) to use partial data
        return await benchmark.run_evaluation(configured_graph, va_list)

    async def _configure_graph(self, dataset, graph, params: dict):
        # Here you can configure the graph based on params
        # For example: set LLM configuration, dataset configuration, etc.
        dataset_config = params.get("dataset", {})
        llm_config = params.get("llm_config", {})
        
        # Instantiate the base workflow
        workflow_instance = graph(name=dataset, llm_config=llm_config, dataset=dataset_config)
        
        # Get task type
        task_type = self.task_types.get(dataset, "math")
        
        # Wrap with MASAFlow ONCE - operators are decorated at this point
        mas_aflow = MASAFlow(workflow_instance, task_type)
        
        # Return the mas_aflow.run method which accepts problem as parameter
        return mas_aflow.run

    def _get_data_path(self, dataset: DatasetType, test: bool) -> str:
        base_path = f"data/datasets/{dataset.lower()}"
        return f"{base_path}_test.jsonl" if test else f"{base_path}_validate.jsonl"
