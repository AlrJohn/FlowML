from typing import Any
from .ast_nodes import (
    Program,
    NumberLiteral,
    UnaryExpression,
    BinaryExpression,
    PrintStatement,
    AssignmentStatement,
    Variable
)
# EVALUATOR
# Walks the AST and computes the result.
# This is the "tree-walking interpreter" — it visits each node and returns

class Evaluator:
    """Visits each AST node and evaluates it to a value."""

    def __init__(self):
        self.variables = {}  # For storing variable values

    def evaluate(self, node: Any) -> int:
        # Dispatch to the right method based on node type
        method_name = f"eval_{type(node).__name__}"
        method = getattr(self, method_name, self.no_eval)
        return method(node)

    def no_eval(self, node: Any):
        raise Exception(f"Evaluator: No eval method for {type(node).__name__}")

    def eval_Program(self, node: Program) -> int:
        results = []
        for stmt in node.statements:
            result = self.evaluate(stmt)
            results.append(result)
        
        return results if results else [] 

    def eval_NumberLiteral(self, node: NumberLiteral) -> int:
        return node.value

    def eval_UnaryExpression(self, node: UnaryExpression) -> int:
        operand = self.evaluate(node.operand)
        if node.operator == '-':
            return -operand
        raise Exception(f"Evaluator: Unknown unary operator '{node.operator}'")

    def eval_BinaryExpression(self, node: BinaryExpression) -> int:
        left = self.evaluate(node.left)
        right = self.evaluate(node.right)
        if node.operator == '+': return left + right
        if node.operator == '-': return left - right
        if node.operator == '*': return left * right
        if node.operator == '/':
            if right == 0:
                raise Exception("Evaluator: Division by zero")
            return left // right    # integer division for now
        raise Exception(f"Evaluator: Unknown operator '{node.operator}'")


    def eval_PrintStatement(self, node: PrintStatement) -> int:
        value = self.evaluate(node.expression)
        print(value)
        return value  # Return the value for testing purposes


    def eval_AssignmentStatement(self, node: AssignmentStatement) -> int:
        value = self.evaluate(node.expression)
        self.variables[node.variable.name] = value
        return value  # Return the assigned value for testing purposes

    def eval_Variable(self, node: Variable) -> int:
        if node.name in self.variables:
            return self.variables[node.name]
        else:
            raise Exception(f"Evaluator: Undefined variable '{node.name}'")
