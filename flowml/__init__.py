from .lexer import Lexer
from .parser import Parser
from .evaluator import Evaluator

def interpret(source: str) -> int:
    """Helper function to run the whole process: lex, parse, evaluate"""
    

    lexer = Lexer(source)

    tokens = lexer.tokenize()
    
    parser = Parser(tokens)

    asts = parser.parse()
    
    results = Evaluator().evaluate(asts)
    return results


