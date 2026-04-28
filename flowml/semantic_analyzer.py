"""
semantic_analyzer.py
--------------------
Static semantic analysis pass for FlowML.

Runs over the AST before execution and collects SemanticErrors for:
  - Use of undeclared variables
  - Type mismatches in assignments (strict mode)
  - Incompatible operand types in arithmetic / comparison expressions
  - ML pipeline ordering violations (e.g. train before model/split)
  - Invalid split ratios (must sum to 1.0)
  - Unknown model names

"""

from __future__ import annotations

from dataclasses import dataclass

from .symbol_table import SymbolTable
from .ast_nodes import (
    Program,
    AssignmentStatement,
    PrintStatement,
    IfStatement,
    WhileStatement,
    BinaryExpression,
    UnaryExpression,
    NumberLiteral,
    FloatLiteral,
    StringLiteral,
    BoolLiteral,
    Variable,
    LoadStatement,
    DropStatement,
    NormalizeStatement,
    SplitStatement,
    ModelStatement,
    TrainStatement,
    EvaluateStatement,
    FunctionDefinition,
    FunctionCall,
    ReturnStatement,
)

# ---------------------------------------------------------------------------
# Known ML models
# ---------------------------------------------------------------------------

KNOWN_MODELS = {
    "LogisticRegression", "RandomForest", "SVM", "KNN", "NaiveBayes",
    "LinearRegression", "Ridge", "Lasso", "KMeans",
}

# Types that can appear in arithmetic expressions
_NUMERIC    = {"int", "float"}
# Types that can be compared with == / !=
_EQUATABLE  = {"int", "float", "str", "bool"}
# Types that support ordered comparisons < > <= >=
_ORDERABLE  = {"int", "float", "str"}


# ---------------------------------------------------------------------------
# SemanticError
# ---------------------------------------------------------------------------

@dataclass
class SemanticError:
    message: str
    node_type: str = "" # AST node class name, for better error messages

    def __str__(self) -> str:
        prefix = f"[{self.node_type}] " if self.node_type else ""
        return f"SemanticError: {prefix}{self.message}"


# ---------------------------------------------------------------------------
# SemanticAnalyzer
# ---------------------------------------------------------------------------

class SemanticAnalyzer:
    """
    Walks the FlowML AST and collects SemanticErrors.

    Uses a SymbolTable (backed by a HashTable) to track declared variables
    and their types. ML pipeline state is tracked via boolean flags so
    ordering violations can be detected without executing any code.
    """

    def __init__(self, strict_types: bool = True):
        self.strict_types = strict_types
        self.symbol_table = SymbolTable(strict_types=False)  # values not needed here
        self.errors: list[SemanticError] = []
        self.functions: dict[str, FunctionDefinition] = {}  # name -> FunctionDefinition node

        # ML pipeline state flags
        self._data_loaded = False
        self._split_done  = False
        self._model_set   = False
        self._trained     = False

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------

    def analyze(self, program: Program) -> list[SemanticError]:
        """Walk the full AST and return all collected SemanticErrors."""
        for stmt in program.statements:
            self._visit(stmt)
        return self.errors

    # Internal helpers
    def _error(self, msg: str, node=None) -> None:
        node_type = type(node).__name__ if node is not None else ""
        self.errors.append(SemanticError(message=msg, node_type=node_type))

    def _visit(self, node) -> str:
        """Dispatch to _visit_<NodeType>; return the inferred type string."""
        method = f"_visit_{type(node).__name__}"
        visitor = getattr(self, method, self._visit_unknown)
        return visitor(node)

    def _visit_unknown(self, node) -> str:
        return "unknown"

    # -----------------------------------------------------------------------
    # Literals and variables
    # -----------------------------------------------------------------------

    def _visit_NumberLiteral(self, node: NumberLiteral) -> str:
        return "int"

    def _visit_FloatLiteral(self, node: FloatLiteral) -> str:
        return "float"

    def _visit_StringLiteral(self, node: StringLiteral) -> str:
        return "str"

    def _visit_BoolLiteral(self, node: BoolLiteral) -> str:
        return "bool"

    def _visit_Variable(self, node: Variable) -> str:
        if not self.symbol_table.contains(node.name):
            self._error(f"Undefined variable '{node.name}'", node)
            return "unknown"
        return self.symbol_table.get_dtype(node.name)

    # -----------------------------------------------------------------------
    # Expressions
    # -----------------------------------------------------------------------

    def _visit_UnaryExpression(self, node: UnaryExpression) -> str:
        operand_type = self._visit(node.operand)
        if operand_type not in _NUMERIC and operand_type != "unknown":
            self._error(
                f"Unary '-' requires a numeric operand, got '{operand_type}'", node
            )
        return operand_type

    def _visit_BinaryExpression(self, node: BinaryExpression) -> str:
        left_type  = self._visit(node.left)
        right_type = self._visit(node.right)
        op = node.operator

        if op in ("==", "!="):
            if left_type != "unknown" and right_type != "unknown":
                if left_type not in _EQUATABLE or right_type not in _EQUATABLE:
                    self._error(
                        f"Operator '{op}' not supported for types "
                        f"'{left_type}' and '{right_type}'", node
                    )
            return "bool"

        if op in ("<", ">", "<=", ">="):
            if left_type != "unknown" and right_type != "unknown":
                if left_type not in _ORDERABLE or right_type not in _ORDERABLE:
                    self._error(
                        f"Operator '{op}' not supported for types "
                        f"'{left_type}' and '{right_type}'", node
                    )
            return "bool"

        if op in ("+", "-", "*", "/"):
            if left_type != "unknown" and right_type != "unknown":
                if left_type not in _NUMERIC or right_type not in _NUMERIC:
                    self._error(
                        f"Operator '{op}' requires numeric operands, "
                        f"got '{left_type}' and '{right_type}'", node
                    )
            if left_type == "float" or right_type == "float":
                return "float"
            return "int"

        return "unknown"

    # -----------------------------------------------------------------------
    # General statements
    # -----------------------------------------------------------------------

    def _visit_AssignmentStatement(self, node: AssignmentStatement) -> None:
        assigned_expression_type = self._visit(node.expression)
        name = node.variable.name

        if self.symbol_table.contains(name):
            declared_type = self.symbol_table.get_dtype(name)
            if (self.strict_types
                    and assigned_expression_type != "unknown"
                    and declared_type != "unknown"
                    and assigned_expression_type != declared_type):
                self._error(
                    f"Type mismatch for '{name}': declared as '{declared_type}', "
                    f"assigned '{assigned_expression_type}'", node
                )
        else:
            dtype = assigned_expression_type if assigned_expression_type != "unknown" else "unknown"
            self.symbol_table.declare_type(name, dtype)

    def _visit_PrintStatement(self, node: PrintStatement) -> None:
        self._visit(node.expression)

    def _visit_IfStatement(self, node: IfStatement) -> None:
        self._visit(node.condition)
        for stmt in node.then_branch:
            self._visit(stmt)
        for stmt in node.else_branch:
            self._visit(stmt)

    def _visit_WhileStatement(self, node: WhileStatement) -> None:
        self._visit(node.condition)
        for stmt in node.body:
            self._visit(stmt)

    def _visit_FunctionDefinition(self, node: FunctionDefinition) -> None:
        """
        Add the function to self.functions so calls can be validated.
        Then analyze the body in a temporary scope where the parameters
        are pre-declared so they don't trigger 'undefined variable' errors.
        """
        # Register function before analyzing body so recursive calls are valid
        self.functions[node.name] = node

        # Temporarily declare parameters in the symbol table
        declared = []
        for param in node.parameters:
            if not self.symbol_table.contains(param):
                self.symbol_table.declare_type(param, "unknown")
                declared.append(param)

        # Analyze the function body
        for stmt in node.body:
            self._visit(stmt)

        # Remove temporarily declared params so they don't
        # pollute the global symbol table after the function definition
        for param in declared:
            try:
                self.symbol_table._table.delete(param)
            except KeyError:
                pass


    def _visit_FunctionCall(self, node: FunctionCall) -> str:
        """
        Validate that the function exists and that the argument count matches.
        Then analyze each argument expression for type/undefined-var errors.
        Returns 'unknown' since we don't track return types.
        """
        if node.name not in self.functions:
            self._error(
                f"Call to undefined function '{node.name}'", node
            )
            # Still analyze arguments even if function is unknown
            for arg in node.arguments:
                self._visit(arg)
            return "unknown"

        func_def = self.functions[node.name]
        expected = len(func_def.parameters)
        got = len(node.arguments)

        if expected != got:
            self._error(
                f"Function '{node.name}' expects {expected} argument(s) "
                f"but was called with {got}",
                node,
            )

        # Analyze each argument expression
        for arg in node.arguments:
            self._visit(arg)

        return "unknown"


    def _visit_ReturnStatement(self, node: ReturnStatement) -> None:
        """
        Analyze the return expression if one exists.
        Bare return (expression is None) is always valid.
        """
        if node.expression is not None:
            self._visit(node.expression)

    # -----------------------------------------------------------------------
    # ML data statements
    # -----------------------------------------------------------------------

    def _visit_LoadStatement(self, node: LoadStatement) -> None:
        self._data_loaded = True

    def _visit_DropStatement(self, node: DropStatement) -> None:
        if not self._data_loaded:
            self._error("'drop' called before any data was loaded", node)

    def _visit_NormalizeStatement(self, node: NormalizeStatement) -> None:
        if not self._data_loaded:
            self._error("'normalize' called before any data was loaded", node)

    def _visit_SplitStatement(self, node: SplitStatement) -> None:
        if not self._data_loaded:
            self._error("'split' called before any data was loaded", node)

        if abs(node.train + node.test - 1.0) > 1e-9:
            self._error(
                f"Split ratios must sum to 1.0 "
                f"(got train={node.train} + test={node.test} = {node.train + node.test:.4f})",
                node,
            )

        for ds_name in ("train_set", "test_set"):
            if not self.symbol_table.contains(ds_name):
                self.symbol_table.declare_type(ds_name, "dataset")

        self._split_done = True

    # -----------------------------------------------------------------------
    # ML model statements
    # -----------------------------------------------------------------------

    def _visit_ModelStatement(self, node: ModelStatement) -> None:
        if node.model_name not in KNOWN_MODELS:
            self._error(
                f"Unknown model '{node.model_name}'. "
                f"Known models: {', '.join(sorted(KNOWN_MODELS))}",
                node,
            )
        self._model_set = True

    def _visit_TrainStatement(self, node: TrainStatement) -> None:
        if not self._model_set:
            self._error("'train' called before a model was defined", node)
        if not self._split_done:
            self._error("'train' called before data was split", node)

        if not self.symbol_table.contains(node.dataset):
            self._error(f"Undefined dataset variable '{node.dataset}'", node)
        elif self.symbol_table.get_dtype(node.dataset) != "dataset":
            self._error(
                f"'{node.dataset}' is not a dataset "
                f"(type: '{self.symbol_table.get_dtype(node.dataset)}')",
                node,
            )

        self._trained = True

    def _visit_EvaluateStatement(self, node: EvaluateStatement) -> None:
        if not self._trained:
            self._error("'evaluate' called before the model was trained", node)

        if not self.symbol_table.contains(node.dataset):
            self._error(f"Undefined dataset variable '{node.dataset}'", node)
        elif self.symbol_table.get_dtype(node.dataset) != "dataset":
            self._error(
                f"'{node.dataset}' is not a dataset "
                f"(type: '{self.symbol_table.get_dtype(node.dataset)}')",
                node,
            )

        if not self.symbol_table.contains(node.result_var):
            self.symbol_table.declare_type(node.result_var, "float")

    # -----------------------------------------------------------------------
    # Program root
    # -----------------------------------------------------------------------

    def _visit_Program(self, node: Program) -> None:
        for stmt in node.statements:
            self._visit(stmt)
