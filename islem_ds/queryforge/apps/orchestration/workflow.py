from dataclasses import dataclass
from typing import List

@dataclass
class WorkflowDefinition:
    name: str
    allowed_tools: List[str]
    max_iterations: int
    budget_tokens: int

WORKFLOWS = {
    "default": WorkflowDefinition(
        name="default",
        allowed_tools=["search_knowledge_base", "calculate", "fetch_external_data"],
        max_iterations=5,
        budget_tokens=4096
    ),
    "safe": WorkflowDefinition(
        name="safe",
        allowed_tools=["search_knowledge_base"],
        max_iterations=3,
        budget_tokens=4096
    )
}
