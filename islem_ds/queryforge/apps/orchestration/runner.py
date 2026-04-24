import json
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from apps.gateway.gateway import LLMGateway
from apps.gateway.providers.base import Message, LLMConfig
from apps.prompts.registry import PromptRegistry
from apps.prompts.engine import TemplateEngine
from apps.prompts.budget import ContextBudgetManager

# wired in INT-1
# from apps.guardrails.pipeline import GuardrailPipeline
# from apps.guardrails.output_guards import OutputGuard

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
    tool_calls_made: list[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    confidence: float = 0.0
    provider_used: str = ""

    def __post_init__(self):
        if self.tool_calls_made is None:
            self.tool_calls_made = []

class WorkflowRunner:
    MAX_TOOL_ITERATIONS = 5

    def __init__(self):
        self._gateway = LLMGateway()
        self._registry = PromptRegistry()
        self._budget = ContextBudgetManager()
        self._engine = TemplateEngine()
        self._dispatcher = ToolDispatcher()

    def run(self, query: str, user: str, session_id: Optional[str] = None, workflow_name: str = "default") -> WorkflowResult:
        # ── DETERMINISTIC ZONE ─────────────────────────────────────────────
        request_id = str(uuid.uuid4())
        start_time = time.monotonic()
        
        # Step 1: Input guard (commented out per instructions)
        # guard_result = InputGuard().check(query)
        
        # Step 2: Load active prompt template
        prompt_version = self._registry.get_active("knowledge_assistant")
        template_body = prompt_version.template_body
        budget_tokens = prompt_version.token_budget
        
        # Step 3: Render the template
        rendered_prompt = self._engine.render(template_body, {"query": query, "user": user})
        
        # Step 4: Enforce context budget
        budget_result = self._budget.fit([Message(role="system", content="You are a helpful assistant."), 
                                         Message(role="user", content=rendered_prompt)], budget_tokens)
        messages = budget_result.messages

        # ── GENERATIVE ZONE ────────────────────────────────────────────────
        iteration = 1
        tool_calls_made = []
        final_answer = ""
        
        # Get schemas that are allowed by the current workflow
        workflow_def = WORKFLOWS.get(workflow_name, WORKFLOWS["default"])
        all_schemas = get_all_schemas()
        allowed_schemas = [s for s in all_schemas if s["function"]["name"] in workflow_def.allowed_tools]
        
        while iteration < self.MAX_TOOL_ITERATIONS:
            config = LLMConfig()  # Use default config
            response = self._gateway.complete(messages, config, query=query)

            # If LLM returns tool_calls -> dispatch
            if response and getattr(response, "tool_calls", None):
                messages.append(Message(role="assistant", content="", tool_calls=response.tool_calls))
                
                for tc in response.tool_calls:
                    tool_calls_made.append(tc.function.name)
                    try:
                        # Parse arguments if they are a JSON string
                        if isinstance(tc.function.arguments, str):
                            args = json.loads(tc.function.arguments)
                        else:
                            args = tc.function.arguments
                        
                        tool_result = self._dispatcher.dispatch(
                            tool_name=tc.function.name,
                            arguments=args,
                            iteration=iteration
                        )
                        result_str = str(tool_result.data) if tool_result.success else tool_result.error
                    except Exception as e:
                        result_str = f"Error: {e}"
                        
                    # Inject results back into conversation
                    messages.append(Message(role="tool", content=result_str, tool_call_id=tc.id))
                
                iteration += 1
            else:
                final_answer = response.content if response else ""
                break

        # ── DETERMINISTIC ZONE (RESUMED) ───────────────────────────────────
        # Output guard (commented out)
        # guard_result = OutputGuard().check(final_answer)
        
        end_time = time.monotonic()
        latency_ms = (end_time - start_time) * 1000
        
        # wired in INT-1 via CostCalculator
        cost_usd = 0.0

        return WorkflowResult(
            answer=final_answer,
            request_id=request_id,
            fallback_used=False,
            truncation_applied=budget_result.truncation_applied,
            turns_removed=budget_result.turns_removed,
            tool_calls_made=tool_calls_made,
            input_tokens=0,  # Would be populated by actual token counting
            output_tokens=0,  # Would be populated by actual token counting
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            confidence=0.0,  # Would be populated by actual confidence scoring
            provider_used=""  # Would be populated by actual provider info
        )
