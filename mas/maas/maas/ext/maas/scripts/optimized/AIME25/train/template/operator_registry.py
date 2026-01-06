from maas.ext.maas.scripts.optimized.AIME25.train.template.operator import (
    Generate,
    GenerateCoT,
    MultiGenerateCoT,
    ScEnsemble,
    Programmer,
    SelfRefine,
    EarlyStop
)

operator_mapping = {
    "Generate": Generate,
    "GenerateCoT": GenerateCoT,
    "MultiGenerateCoT": MultiGenerateCoT,
    "ScEnsemble": ScEnsemble,
    "Programmer": Programmer,
    "SelfRefine": SelfRefine,
    "EarlyStop": EarlyStop,
}

operator_names = list(operator_mapping.keys())
