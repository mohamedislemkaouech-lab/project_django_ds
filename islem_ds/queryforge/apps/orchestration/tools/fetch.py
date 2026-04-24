from .base import BaseTool, ToolResult

MOCK_DATA = {
    "weather": lambda topic: f"The weather for {topic} is sunny and 75°F.",
    "news": lambda topic: f"Breaking news about {topic}: It's going great!",
}

class FetchExternalDataTool(BaseTool):
    name = "fetch_external_data"
    description = "Fetches external data such as weather or news."
    parameters = {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "The topic to fetch data for."
            },
            "source": {
                "type": "string",
                "enum": ["weather", "news"],
                "description": "The source of the data."
            }
        },
        "required": ["topic"]
    }

    def execute(self, topic: str, source: str = "news") -> ToolResult:
        try:
            if source not in MOCK_DATA:
                raise ValueError(f"Unknown source: {source}")
            data = MOCK_DATA[source](topic)
            return ToolResult(success=True, data={"result": data})
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
