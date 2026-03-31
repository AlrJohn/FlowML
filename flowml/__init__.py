from .lexer import Lexer
from .parser import Parser
from .evaluator import Evaluator
from .environment import Environment, ReturnException
from .semantic_analyzer import SemanticAnalyzer, SemanticError

def interpret(source: str) -> int:
    """Helper function to run the whole process: lex, parse, evaluate"""
    lexer  = Lexer(source)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    asts   = parser.parse()
    return Evaluator().evaluate(asts)

def analyze(source: str, strict_types: bool = True) -> list:
    """Lex, parse, and semantically analyze "source". Returns a list of SemanticErrors."""
    tokens = Lexer(source).tokenize()
    ast    = Parser(tokens).parse()
    return SemanticAnalyzer(strict_types=strict_types).analyze(ast)


