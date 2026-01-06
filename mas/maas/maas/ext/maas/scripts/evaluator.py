from typing import Dict, Literal, Tuple
from maas.ext.maas.benchmark.benchmark import BaseBenchmark
from maas.ext.maas.benchmark.aime import AIMEBenchmark
from maas.ext.maas.benchmark.gaia import GAIABenchmark

DatasetType = Literal["AIME24", "AIME25", "GAIA"]


class Evaluator:
    def __init__(self, eval_path: str, batch_size: int):
        self.eval_path = eval_path
        self.batch_size = batch_size
        self.dataset_configs: Dict[DatasetType, BaseBenchmark] = {
            "AIME24": AIMEBenchmark,
            "AIME25": AIMEBenchmark,
            "GAIA": GAIABenchmark
        }

    async def graph_evaluate(
        self, dataset: DatasetType, graph, params: dict, path: str, is_test: bool = False
    ):
        if dataset not in self.dataset_configs:
            raise ValueError(f"Unsupported dataset: {dataset}")

        data_path = self._get_data_path(dataset, is_test)
        benchmark_class = self.dataset_configs[dataset]

        benchmark = benchmark_class(
            name=dataset, 
            file_path=data_path, 
            log_path=path, 
            batch_size=self.batch_size,
            controller=params["controller"],
            operator_embeddings=params["operator_embeddings"],
            optimizer=params["optimizer"],
        )
        configured_graph = await self._configure_graph(dataset, graph, params)
        if is_test:
            va_list = [0]
        else:
            va_list = None
        return await benchmark.run_evaluation(configured_graph, va_list, is_test, params["sample"], params["is_textgrad"])

    async def _configure_graph(self, dataset, graph, params: dict):
        controller = params.get("controller")
        operator_embeddings = params.get("operator_embeddings")
        llm_config = params.get("execute_llm_config")
        dataset_config = params.get("dataset")
        configured_graph = graph(
            name=dataset,
            llm_config=llm_config,
            dataset=dataset_config,
            controller=controller,
            operator_embeddings=operator_embeddings,
        )
        return configured_graph

    def _get_data_path(self, dataset: DatasetType, test: bool) -> str:
        base_path = f"maas/ext/maas/data/{dataset.lower()}"
        return f"{base_path}_test.jsonl" if test else f"{base_path}_train.jsonl" 
