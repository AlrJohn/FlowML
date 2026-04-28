from .lexer import Lexer
from .parser import Parser
from .evaluator import Evaluator
from .environment import Environment, ReturnException
from .semantic_analyzer import SemanticAnalyzer, SemanticError
from .codegen import CodeGenerator


def interpret(source: str):
    """Lex, parse, semantically analyze, and evaluate a FlowML program."""
    lexer  = Lexer(source)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast    = parser.parse()
    return Evaluator().evaluate(ast)


def analyze(source: str, strict_types: bool = True) -> list:
    """Lex, parse, and semantically analyze source. Returns list of SemanticErrors."""
    tokens = Lexer(source).tokenize()
    ast    = Parser(tokens).parse()
    return SemanticAnalyzer(strict_types=strict_types).analyze(ast)


def compile_to_python(source: str) -> str:
    """
    Lex, parse, and compile a FlowML program to Python source code.
    Returns the generated Python as a string.
    Raises SemanticError if the program has semantic errors.
    """
    tokens = Lexer(source).tokenize()
    ast    = Parser(tokens).parse()

    # Run semantic analysis before generating code
    errors = SemanticAnalyzer(strict_types=False).analyze(ast)
    if errors:
        for err in errors:
            print(err)
        raise Exception("Compilation aborted due to semantic errors.")

    return CodeGenerator().generate(ast)