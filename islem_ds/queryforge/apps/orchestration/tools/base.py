from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

@dataclass
class ToolResult:
    success: bool
    data: Any
    error: str = ""

class BaseTool(ABC):
    name: str = ""
    description: str = ""
    parameters: dict = {}

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        pass

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }
