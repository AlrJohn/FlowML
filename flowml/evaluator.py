from typing import Any
import numpy as np

import pandas as pd

from .MLBackend import MLBackend
from .environment import Environment, ReturnException

from .ast_nodes import (
    EvaluateStatement,
    FloatLiteral,
    FunctionCall,
    FunctionDefinition,
    IfStatement,
    ModelStatement,
    ReturnStatement,
    StringLiteral,
    BoolLiteral,
    TrainStatement,
    WhileStatement,
    Program,
    NumberLiteral,
    UnaryExpression,
    BinaryExpression,
    PrintStatement,
    LoadStatement,
    DropStatement,
    NormalizeStatement,
    SplitStatement,
    AssignmentStatement,
    Variable
)
# EVALUATOR
# Walks the AST and computes the result.
# This is the "tree-walking interpreter" — it visits each node and returns
# the result of evaluating that node.

# Global state (like variables) is stored in the Evaluator instance.
#the result from loadStatement is a pandas DataFrame, which we can store in the variables dictionary just like any other value. This allows us to manipulate the DataFrame in subsequent statements using its variable name.

class Evaluator:
    """Visits each AST node and evaluates it to a value."""

    def __init__(self):
        self.env = Environment() # Environment for storing variable values
        self.ml_backend = MLBackend()  # Instance of MLBackend for ML operations

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

    def eval_IfStatement(self, node: IfStatement) -> Any:
        condition = self.evaluate(node.condition)
        if condition:
            result = None
            for stmt in node.then_branch:
                result = self.evaluate(stmt)
            return result
        elif node.else_branch:
            result = None
            for stmt in node.else_branch:
                result = self.evaluate(stmt)
            return result
        return None

    def eval_WhileStatement(self, node: WhileStatement) -> Any:
        """
        Evaluate a while loop.
        Re-evaluates the condition before each iteration.
        Terminates when condition is false.
        Returns the result of the last statement in the last iteration,
        or None if the loop body never executed.
        """
        result = None
        while self.evaluate(node.condition):
            for stmt in node.body:
                result = self.evaluate(stmt)
        return result
        

    def eval_FunctionDefinition(self, node: FunctionDefinition):
        """
        Store the function definition in the global environment
        so it can be called from anywhere in the program.
        The FunctionDefinition node itself is the stored value.
        """
        self.env.set_global(node.name, node)
        return None



    def eval_FunctionCall(self, node: FunctionCall):
        """
        Call a function:
        1. Look up the function definition by name
        2. Evaluate all argument expressions in the CURRENT scope
        3. Create a new child Environment for the function's local scope
        4. Bind parameter names to argument values in the child scope
        5. Execute the function body statements
        6. Catch ReturnException and return its value
        7. Return None if no return statement was hit
        """
        # Step 1 — look up function definition
        func_def = self.env.get(node.name)
        if not isinstance(func_def, FunctionDefinition):
            raise Exception(f"'{node.name}' is not a function")

        # Step 2 — evaluate arguments in current scope
        if len(node.arguments) != len(func_def.parameters):
            raise Exception(
                f"Function '{node.name}' expects {len(func_def.parameters)} "
                f"argument(s) but got {len(node.arguments)}"
            )
        arg_values = [self.evaluate(arg) for arg in node.arguments]

        # Step 3 — create new child scope
        previous_env = self.env
        self.env = Environment(parent=previous_env)

        # Step 4 — bind params to args in new scope
        for param, value in zip(func_def.parameters, arg_values):
            self.env.set(param, value)

        # Step 5 & 6 — execute body, catch return
        result = None
        try:
            for stmt in func_def.body:
                self.evaluate(stmt)
        except ReturnException as r:
            result = r.value
        finally:
            # Step 7 — always restore previous scope
            self.env = previous_env

        return result


    def eval_ReturnStatement(self, node: ReturnStatement) -> Any:

        value = self.evaluate(node.expression) if node.expression else None
        raise ReturnException(value)


    def eval_NumberLiteral(self, node: NumberLiteral) -> int:
        return node.value
    
    def eval_FloatLiteral(self, node: FloatLiteral) -> float:
        return node.value

    def eval_StringLiteral(self, node: StringLiteral) -> str:
        return node.value

    def eval_BoolLiteral(self, node: BoolLiteral) -> bool:
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
        if node.operator == '==': return left == right
        if node.operator == '!=': return left != right
        if node.operator == '<':  return left < right
        if node.operator == '>':  return left > right
        if node.operator == '<=': return left <= right
        if node.operator == '>=': return left >= right
        raise Exception(f"Evaluator: Unknown operator '{node.operator}'")


    def eval_PrintStatement(self, node: PrintStatement) -> int:
        value = self.evaluate(node.expression)
        if node.newline:
            print(value, end='\n')
        else:
            print(value, end='')
        return value  # Return the value for testing purposes

    def eval_LoadStatement(self, node: LoadStatement) -> pd.DataFrame:
        self.ml_backend.load(node.filename)
        return self.ml_backend.df  # Return the DataFrame for testing purposes

    def eval_DropStatement(self, node: DropStatement) -> pd.DataFrame:
        if self.ml_backend.df is None:
            raise Exception("Evaluator: No active DataFrame to drop column from")
        self.ml_backend.drop(node.column_names)
        return self.ml_backend.df  # Return the DataFrame for testing purposes
    
    def eval_NormalizeStatement(self, node: NormalizeStatement) -> pd.DataFrame:
        #we just store the columns to be normalized, and wait for the train statement to actually perform the normalization. This allows us to keep the normalization logic separate from the statement that specifies which columns to normalize.
        if self.ml_backend.df is None:
            raise Exception("Evaluator: No active DataFrame to normalize")

        self.ml_backend.normalize(node.column_names)
        return self.ml_backend.df  # Return the DataFrame for testing purposes
    
    def eval_SplitStatement(self, node: SplitStatement) -> tuple:
        
        self.ml_backend.split(node.train, node.test)
        # bind results as regular variables so they can be referenced by name
        self.env.set_global('train_set', self.ml_backend.train_set) # bind the train set as a global variable
        self.env.set_global('test_set', self.ml_backend.test_set)  # bind the test set as a global variable

    def eval_ModelStatement(self, node: ModelStatement) -> str:
        self.ml_backend.set_model(node.model_name, node.params)

    def eval_TrainStatement(self, node: TrainStatement) -> str:
        dataset = self.env.get(node.dataset)
        self.ml_backend.train(dataset)

    def eval_EvaluateStatement(self, node: EvaluateStatement):
        dataset = self.env.get(node.dataset)
        result  = self.ml_backend.evaluate(dataset)
        self.env.set(node.result_var, result)
        return result 

    def eval_AssignmentStatement(self, node: AssignmentStatement) -> int:
        value = self.evaluate(node.expression)
        self.env.set(node.variable.name, value)
        return value  # Return the assigned value for testing purposes

    def eval_Variable(self, node: Variable) -> int:
        return self.env.get(node.name)
