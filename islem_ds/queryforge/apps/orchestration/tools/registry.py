from typing import Dict, List, Any
from .base import BaseTool, ToolResult

_REGISTRY: Dict[str, BaseTool] = {}

class ToolNotFoundError(Exception):
    """Raised when a requested tool is not found in the registry."""
    pass

class ToolLoopError(Exception):
    """Raised when the maximum tool iterations are exceeded."""
    pass

def register(tool: BaseTool) -> None:
    """Registers a tool in the global registry."""
    _REGISTRY[tool.name] = tool

def get_all_schemas() -> List[dict]:
    """Returns OpenAI-format schemas for all registered tools."""
    return [tool.to_openai_schema() for tool in _REGISTRY.values()]

class ToolDispatcher:
    MAX_ITERATIONS = 5

    def dispatch(self, tool_name: str, arguments: dict, iteration: int) -> ToolResult:
        if iteration > self.MAX_ITERATIONS:
            raise ToolLoopError(f"Maximum tool iterations ({self.MAX_ITERATIONS}) exceeded.")

        if tool_name not in _REGISTRY:
            raise ToolNotFoundError(f"Tool '{tool_name}' not found.")

        tool = _REGISTRY[tool_name]
        try:
            return tool.execute(**arguments)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
