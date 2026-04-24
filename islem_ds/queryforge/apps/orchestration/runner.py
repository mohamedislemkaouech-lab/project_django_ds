from dataclasses import dataclass
from typing import Optional

# Mocking the imports for testing purposes if they don't exist yet
try:
    from apps.gateway.gateway import LLMGateway
    from apps.gateway.providers.base import Message, LLMConfig
except ImportError:
    class LLMGateway:
        def complete(self, *args, **kwargs): pass
    class Message:
        def __init__(self, role, content, tool_calls=None):
            self.role = role
            self.content = content
            self.tool_calls = tool_calls
    class LLMConfig: pass

try:
    from apps.prompts.registry import PromptRegistry
    from apps.prompts.engine import TemplateEngine
    from apps.prompts.budget import ContextBudgetManager
except ImportError:
    class PromptRegistry:
        def get_active_template(self, name): return "template"
    class TemplateEngine:
        def render(self, template, context): return f"{template} rendered {context}"
    class ContextBudgetManager:
        def enforce_budget(self, text, budget): return text

from .tools.registry import ToolDispatcher, get_all_schemas
from .tools.search import SearchKnowledgeBaseTool
from .tools.calculator import CalculatorTool
from .tools.fetch import FetchExternalDataTool
from .workflow import WorkflowDefinition, WORKFLOWS

# Ensure tools are registered
from .tools.registry import register
register(SearchKnowledgeBaseTool())
register(CalculatorTool())
register(FetchExternalDataTool())


@dataclass
class WorkflowResult:
    answer: str
    request_id: str = ""
    fallback_used: bool = False
    truncation_applied: bool = False
    turns_removed: int = 0
    tool_calls_made: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    confidence: float = 0.0
    provider_used: str = ""

class WorkflowRunner:
    MAX_TOOL_ITERATIONS = 5

    def __init__(self):
        self._gateway = LLMGateway()
        self._registry = PromptRegistry()
        self._budget = ContextBudgetManager()
        self._engine = TemplateEngine()
        self._dispatcher = ToolDispatcher()

    def run(self, query: str, user: str, session_id: Optional[str] = None, workflow_name: str = "default") -> WorkflowResult:
        # ── DETERMINISTIC ZONE ──
        # Step 1: Input guard (commented out per instructions)
        # guard_result = InputGuard().check(query)
        
        # Step 2: Load active prompt template
        template = self._registry.get_active_template("default")
        
        # Step 3: Render the template
        rendered_prompt = self._engine.render(template, {"query": query, "user": user})
        
        # Step 4: Enforce context budget
        workflow_def = WORKFLOWS.get(workflow_name, WORKFLOWS["default"])
        safe_prompt = self._budget.enforce_budget(rendered_prompt, workflow_def.budget_tokens)

        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content=safe_prompt)
        ]

        # ── GENERATIVE ZONE ──
        iteration = 0
        tool_calls_count = 0
        final_answer = ""
        
        # Get schemas that are allowed by the current workflow
        all_schemas = get_all_schemas()
        allowed_schemas = [s for s in all_schemas if s["function"]["name"] in workflow_def.allowed_tools]
        
        while iteration <= self.MAX_TOOL_ITERATIONS:
            response = self._gateway.complete(
                messages=messages,
                tools=allowed_schemas if allowed_schemas else None
            )

            # If LLM returns tool_calls -> dispatch
            if response and getattr(response, "tool_calls", None):
                messages.append(Message(role="assistant", content="", tool_calls=response.tool_calls))
                
                for tool_call in response.tool_calls:
                    tool_calls_count += 1
                    try:
                        tool_result = self._dispatcher.dispatch(
                            tool_name=tool_call.name,
                            arguments=tool_call.arguments,
                            iteration=iteration
                        )
                        result_str = str(tool_result.data) if tool_result.success else tool_result.error
                    except Exception as e:
                        result_str = f"Error: {e}"
                        
                    # Inject results back into conversation
                    # Usually role='tool' or 'function' depending on API, assuming 'tool' here.
                    messages.append(Message(role="tool", content=result_str))
                
                iteration += 1
            else:
                final_answer = response.content if response else ""
                break

        # ── DETERMINISTIC ZONE (RESUMED) ──
        # Output guard (commented out)
        # guard_result = OutputGuard().check(final_answer)

        return WorkflowResult(
            answer=final_answer,
            tool_calls_made=tool_calls_count,
            latency_ms=150.0  # mock latency
        )
