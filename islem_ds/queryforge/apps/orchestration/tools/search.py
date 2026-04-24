from .base import BaseTool, ToolResult
from apps.orchestration.models import KnowledgeItem
from django.core.serializers import serialize
import json

class SearchKnowledgeBaseTool(BaseTool):
    name = "search_knowledge_base"
    description = "Searches a database of knowledge items for relevant information."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to match against knowledge item content."
            },
            "top_k": {
                "type": "integer",
                "description": "The maximum number of results to return.",
                "default": 3
            }
        },
        "required": ["query"]
    }

    def execute(self, query: str, top_k: int = 3) -> ToolResult:
        try:
            results = KnowledgeItem.objects.filter(content__icontains=query)[:top_k]
            # Convert QuerySet to a format suitable for the tool's data
            data = [{"title": item.title, "content": item.content, "source": item.source} for item in results]
            return ToolResult(success=True, data=data)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
