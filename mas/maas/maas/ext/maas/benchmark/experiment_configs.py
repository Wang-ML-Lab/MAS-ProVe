from typing import Dict, List

class ExperimentConfig:
    def __init__(self, dataset: str, question_type: str, operators: List[str]):
        self.dataset = dataset
        self.question_type = question_type
        self.operators = operators

EXPERIMENT_CONFIGS: Dict[str, ExperimentConfig] = {
    "AIME24": ExperimentConfig(
        dataset="AIME24",
        question_type="math",
        operators=["Generate", "GenerateCoT", "MultiGenerateCoT", "ScEnsemble", "Programmer", "SelfRefine", "EarlyStop"],
    ),
    "AIME25": ExperimentConfig(
        dataset="AIME25",
        question_type="math",
        operators=["Generate", "GenerateCoT", "MultiGenerateCoT", "ScEnsemble", "Programmer", "SelfRefine", "EarlyStop"],
    ),
    "GAIA": ExperimentConfig(
        dataset="GAIA",
        question_type="qa",
        operators=["Generate", "GenerateCoT", "MultiGenerateCoT", "ScEnsemble", "SelfRefine", "EarlyStop"],
    )
}
