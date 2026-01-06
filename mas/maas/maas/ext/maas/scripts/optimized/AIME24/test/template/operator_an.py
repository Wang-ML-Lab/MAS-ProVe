from pydantic import BaseModel, Field


class GenerateOp(BaseModel):
    response: str = Field(default="", description="Your solution for this problem")

class CodeGenerateOp(BaseModel):
    code: str = Field(default="", description="Your complete code solution for this problem")

class ScEnsembleOp(BaseModel):
    # ADD THIS FIELD to match your prompt's instruction
    thought: str = Field(default="", description="The step-by-step reasoning comparing the solutions.")
    final_answer: str = Field(default="", description="The numeric answer of most consistent solution.")

class SelfRefineOp(BaseModel):
    response: str = Field(default="", description="Your solution for this problem")
