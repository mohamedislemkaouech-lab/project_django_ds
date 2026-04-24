import ast
import operator
from .base import BaseTool, ToolResult

# Safe math operators
ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

def safe_eval(node):
    if isinstance(node, ast.Expression):
        return safe_eval(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.BinOp):
        left = safe_eval(node.left)
        right = safe_eval(node.right)
        op_type = type(node.op)
        if op_type in ALLOWED_OPS:
            return ALLOWED_OPS[op_type](left, right)
        raise ValueError(f"Unsupported operator: {op_type}")
    elif isinstance(node, ast.UnaryOp):
        operand = safe_eval(node.operand)
        op_type = type(node.op)
        if op_type in ALLOWED_OPS:
            return ALLOWED_OPS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type}")
    else:
        raise ValueError(f"Unsupported AST node: {type(node)}")

class CalculatorTool(BaseTool):
    name = "calculate"
    description = "Safely evaluates a mathematical expression."
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate (e.g., '(3 + 4) * 2')."
            }
        },
        "required": ["expression"]
    }

    def execute(self, expression: str) -> ToolResult:
        try:
            tree = ast.parse(expression, mode="eval")
            result = safe_eval(tree)
            return ToolResult(success=True, data={"result": result})
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
