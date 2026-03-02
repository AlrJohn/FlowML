# AST NODES
# The tree structure that represents the meaning of the source code.
# Each node type is one kind of construct in the language.

from dataclasses import dataclass
from typing import List, Any

@dataclass
class Variable:
    """Represents a variable in the AST"""
    name: str

@dataclass
class AssignmentStatement:
    """Represents an assignment statement in the AST"""
    variable: Variable
    expression: Any

@dataclass
class PrintStatement:
    """Represents a print statement in the AST"""
    expression: Any

@dataclass
class NumberLiteral:
    """Represents a number literal in the AST"""
    value: int

@dataclass
class UnaryExpression:
    """Represents a unary expression in the AST"""
    operator: str
    operand: Any

@dataclass
class BinaryExpression:
    """Represents a binary expression in the AST"""
    left: Any
    operator: str
    right: Any

@dataclass
class Program:
    """Represents the entire program (a sequence of statements)"""
    statements: List[Any]