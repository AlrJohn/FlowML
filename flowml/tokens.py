from enum import Enum, auto
from dataclasses import dataclass
from typing import Any

class TokenType(Enum):
    """All possible token types in our language"""
    # Literals
    NUMBER = auto() # Integer literals
    
    # Keywords
    PRINT = auto()

    # Identifiers (variable names, function names, etc.)
    IDENTIFIER = auto()

    ASSIGN = auto() # =
    
    # Operators

    PLUS = auto() # the auto function automatically assigns values to the enum members
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    
    
    # Punctuation
    LPAREN = auto()      # (
    RPAREN = auto()      # )
    SEMICOLON = auto()   # ;
    
    # Special
    EOF = auto()         # End of file


@dataclass
class Token:
    """Represents a single token"""
    type: TokenType
    value: Any
    line: int
    column: int
    
    def __repr__(self):
        return f"Token({self.type.name}, {self.value}, {self.line}:{self.column})"

