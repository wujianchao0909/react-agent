import ast
import math
import operator
from typing import Annotated
from pydantic import Field
from langchain_core.tools import tool

# 允许的二元运算符
BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv
}

# 允许的一元运算符
UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg
}

# 允许的数学函数
SAFE_FUNCS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "pow": math.pow,
    "floor": math.floor,
    "ceil": math.ceil
}

SAFE_CONSTS = {"pi": math.pi, "e": math.e}

def eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in BIN_OPS:
        return BIN_OPS[type(node.op)](eval_node(node.left), eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPS:
        return UNARY_OPS[type(node.op)](eval_node(node.operand))
    if isinstance(node, ast.Name):
        if node.id in SAFE_FUNCS:
            return SAFE_FUNCS[node.id]
        raise ValueError(f"未允许的变量：{node.id}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_FUNCS:
            raise ValueError(f"未允许的函数：{getattr(node.func, 'id', node.func)}")
        if node.keywords:
            raise ValueError("不支持关键字参数")
        return SAFE_FUNCS[node.func.id](*[eval_node(a) for a in node.args])
    raise ValueError(f"不支持的语法：{type(node).__name__}")

@tool
def calculator(expression: Annotated[str, Field(description='数学表达式，如 "2 + 3 * 4" 或 "sqrt(16)"')]) -> str:
    """执行数学计算。当用户需要做算术运算时使用。"""
    try:
        tree = ast.parse(expression, mode="eval")
        result = eval_node(tree)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算失败：{e}。请检查表达式格式。"



