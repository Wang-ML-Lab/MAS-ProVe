from maas.ext.maas.scripts.optimized.GAIA.train.template.operator import (
    Generate,
    GenerateCoT,
    MultiGenerateCoT,
    ScEnsemble,
    SelfRefine,
    EarlyStop
)

operator_mapping = {
    "Generate": Generate,
    "GenerateCoT": GenerateCoT,
    "MultiGenerateCoT": MultiGenerateCoT,
    "ScEnsemble": ScEnsemble,
    "SelfRefine": SelfRefine,
    "EarlyStop": EarlyStop,
}

operator_names = list(operator_mapping.keys())
