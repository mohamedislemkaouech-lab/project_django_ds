import unittest
from unittest.mock import patch, MagicMock
from apps.orchestration.tools.base import BaseTool, ToolResult
from apps.orchestration.tools.registry import ToolDispatcher, ToolNotFoundError, ToolLoopError, _REGISTRY, register
from apps.orchestration.tools.calculator import CalculatorTool
from apps.orchestration.tools.fetch import FetchExternalDataTool
from apps.orchestration.runner import WorkflowRunner, WorkflowResult

# For Django tests
from django.test import TestCase
from apps.orchestration.models import KnowledgeItem
from apps.orchestration.tools.search import SearchKnowledgeBaseTool

class TestOrchestration(unittest.TestCase):

    def setUp(self):
        # Clear registry for clean tests (if not relying on global loading)
        pass

    def test_calculator_tool_evaluates_expression(self):
        tool = CalculatorTool()
        result = tool.execute(expression="2 + 2")
        self.assertTrue(result.success)
        self.assertEqual(result.data["result"], 4)
        
        result2 = tool.execute(expression="(3 + 4) * 2")
        self.assertTrue(result2.success)
        self.assertEqual(result2.data["result"], 14)

    def test_calculator_tool_blocks_code_injection(self):
        tool = CalculatorTool()
        result = tool.execute(expression="__import__('os')")
        self.assertFalse(result.success)
        self.assertIn("Unsupported AST node", result.error)

    def test_tool_dispatcher_raises_for_unknown_tool(self):
        dispatcher = ToolDispatcher()
        with self.assertRaises(ToolNotFoundError):
            dispatcher.dispatch("unknown_tool", {}, 1)

    def test_tool_dispatcher_raises_tool_loop_error_at_max(self):
        dispatcher = ToolDispatcher()
        tool = CalculatorTool()
        _REGISTRY["calculate"] = tool
        with self.assertRaises(ToolLoopError):
            dispatcher.dispatch("calculate", {"expression": "2+2"}, 6)

    def test_tool_that_raises_returns_failed_tool_result(self):
        class CrashingTool(BaseTool):
            name = "crasher"
            def execute(self, **kwargs):
                raise ValueError("Oops I crashed")
        
        tool = CrashingTool()
        _REGISTRY["crasher"] = tool
        dispatcher = ToolDispatcher()
        result = dispatcher.dispatch("crasher", {}, 1)
        
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Oops I crashed")

    @patch('apps.orchestration.runner.LLMGateway')
    def test_workflow_runner_returns_result_without_tools(self, mock_gateway_class):
        mock_gateway = mock_gateway_class.return_value
        
        # Mock successful LLM response without tools
        mock_response = MagicMock()
        mock_response.content = "This is a basic answer."
        mock_response.tool_calls = None
        mock_gateway.complete.return_value = mock_response

        runner = WorkflowRunner()
        # Ensure gateway mock is applied
        runner._gateway = mock_gateway
        
        result = runner.run("Hello", "User1")
        self.assertEqual(result.answer, "This is a basic answer.")
        self.assertEqual(result.tool_calls_made, 0)

    @patch('apps.orchestration.runner.LLMGateway')
    @patch('apps.orchestration.tools.registry.ToolDispatcher.dispatch')
    def test_workflow_runner_dispatches_tool_and_loops(self, mock_dispatch, mock_gateway_class):
        mock_gateway = mock_gateway_class.return_value
        
        # Turn 1: LLM returns a tool call
        mock_response_1 = MagicMock()
        mock_response_1.content = ""
        mock_tool_call = MagicMock()
        mock_tool_call.name = "calculate"
        mock_tool_call.arguments = {"expression": "2+2"}
        mock_response_1.tool_calls = [mock_tool_call]
        
        # Turn 2: LLM returns final answer
        mock_response_2 = MagicMock()
        mock_response_2.content = "The answer is 4."
        mock_response_2.tool_calls = None
        
        mock_gateway.complete.side_effect = [mock_response_1, mock_response_2]
        
        # Tool returns successful result
        mock_dispatch.return_value = ToolResult(success=True, data={"result": 4})
        
        runner = WorkflowRunner()
        runner._gateway = mock_gateway
        
        result = runner.run("What is 2+2?", "User1")
        
        self.assertEqual(result.answer, "The answer is 4.")
        self.assertEqual(result.tool_calls_made, 1)
        mock_dispatch.assert_called_once_with(tool_name="calculate", arguments={"expression": "2+2"}, iteration=0)

class TestKnowledgeBaseTool(TestCase):
    
    def setUp(self):
        KnowledgeItem.objects.create(title="Django Tips", content="Django ORM is powerful.", source="Book")
        KnowledgeItem.objects.create(title="Python Info", content="Python lists are mutable.", source="Web")
        register(SearchKnowledgeBaseTool())
        
    def test_knowledge_item_search_returns_matching_results(self):
        tool = SearchKnowledgeBaseTool()
        result = tool.execute(query="Django")
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.data), 1)
        self.assertEqual(result.data[0]["title"], "Django Tips")
