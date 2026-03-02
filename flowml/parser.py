from typing import List, Any
from .tokens import Token, TokenType
from .ast_nodes import (
    Variable,
    AssignmentStatement,
    PrintStatement,
    NumberLiteral,
    UnaryExpression,
    BinaryExpression,
    Program
)


# PARSER
# Consumes a flat list of tokens and produces an AST.
#
# Operator precedence (low -> high):
#   +  -          (parse_expression)
#   *  /          (parse_term)
#   unary -       (parse_unary)
#   number, ()    (parse_factor)


class Parser:
    """Parses tokens into an abstract syntax tree (AST)"""
    # For simplicity, we won't implement the parser in this example

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def current_token(self) -> Token:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None
    
    def eat(self, token_type: TokenType) -> Token:
        """Consume a token of the expected type, or raise an error"""
        token = self.current_token()
        if token and token.type == token_type:
            self.pos += 1
            return token
        else:
            raise Exception(
            f"Parser Error at {token.line}:{token.column}: "
            f"Expected {token_type.name} but got {token.type.name} ({token.value!r})"
        )
    
    def parse(self) -> List[Any]:
        """Parse the tokens into an AST"""

        nodes = self.parse_program()
        self.eat(TokenType.EOF)  # Ensure we consumed all tokens
        return nodes
    
    def parse_statement(self) -> Any:
        """Parse a single statement (like 'print expression;')"""
        if self.current_token().type == TokenType.PRINT:
            self.eat(TokenType.PRINT)
            expr = self.parse_expression()
            self.eat(TokenType.SEMICOLON)
            return PrintStatement(expression=expr)
        
        elif self.current_token().type == TokenType.IDENTIFIER:
            if self.peek_token().type == TokenType.ASSIGN:  # look ahead
                var_token = self.eat(TokenType.IDENTIFIER)
                self.eat(TokenType.ASSIGN)
                expr = self.parse_expression()
                self.eat(TokenType.SEMICOLON)
                return AssignmentStatement(variable=Variable(name=var_token.value), expression=expr)
            else:
                expr = self.parse_expression()  # treat as expression
                self.eat(TokenType.SEMICOLON)
                return expr
        
        else:
            expr = self.parse_expression()
            self.eat(TokenType.SEMICOLON)
            return expr
        

        # else:
        #     raise Exception(
        #         f"Parser Error at {self.current_token().line}:{self.current_token().column}: "
        #         f"Unexpected token {self.current_token().type.name} ({self.current_token().value!r})"
        #     )
        
    def parse_program(self) -> Program:
        """Parse a sequence of statements until EOF"""
        statements = []
        while self.current_token().type != TokenType.EOF:
            stmt = self.parse_statement()
            statements.append(stmt)
        return Program(statements=statements)


    def parse_expression(self) -> Any:
        """Parse an expression (handles + and -)
            expression = term (('+' | '-') term)*
            """
        node = self.parse_term()
        
        while self.current_token() and self.current_token().type in (TokenType.PLUS, TokenType.MINUS):
            op_token = self.eat(self.current_token().type) # Consume the operator. already moves pos forward
            right = self.parse_term()
            node = BinaryExpression(left=node, operator=op_token.value, right=right)
        
        return node
    
    def parse_term(self) -> Any:
        """
        Handles * and / (higher precedence than + and -).
        term = unary (('*' | '/') unary)*
        """
        node = self.parse_unary()
        
        while self.current_token() and self.current_token().type in (TokenType.MULTIPLY, TokenType.DIVIDE):
            op_token = self.eat(self.current_token().type) # Consume the operator. already moves pos forward
            right = self.parse_unary()
            node = BinaryExpression(left=node, operator=op_token.value, right=right)
        
        return node
    
    def parse_unary(self) -> Any:
        """
        Handles unary operators (like -).
        unary = ('-' unary) | factor
        """
        if self.current_token() and self.current_token().type == TokenType.MINUS:
            op_token = self.eat(TokenType.MINUS) # Consume the operator. already moves pos forward
            operand = self.parse_unary()
            return UnaryExpression(operator=op_token.value, operand=operand)
        
        return self.parse_factor()
    
    def parse_factor(self) -> Any:
        """
        Handles the highest-precedence things: numbers and parenthesized expressions.
        factor = NUMBER | '(' expression ')'
        """
        token = self.current_token()
        
        if token.type == TokenType.NUMBER:
            self.eat(TokenType.NUMBER)
            return NumberLiteral(value=token.value)
        
        elif token.type == TokenType.IDENTIFIER:
            self.eat(TokenType.IDENTIFIER)
            return Variable(name=token.value)
        
        elif token.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            node = self.parse_expression()
            self.eat(TokenType.RPAREN)
            return node
        
        else:
            raise Exception(
                f"Parser Error at {token.line}:{token.column}: "
                f"Unexpected token {token.type.name} ({token.value!r})"
            )
        
    def peek_token(self) -> Token:
        if self.pos + 1 < len(self.tokens):
            return self.tokens[self.pos + 1]
        return None

