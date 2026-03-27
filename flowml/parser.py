from typing import List, Any
from .tokens import Token, TokenType
from .ast_nodes import (
    EvaluateStatement,
    FloatLiteral,
    IfStatement,
    StringLiteral,
    TrainStatement,
    WhileStatement,
    Variable,
    AssignmentStatement,
    PrintStatement,
    NumberLiteral,
    UnaryExpression,
    BinaryExpression,
    BoolLiteral,
    LoadStatement,
    DropStatement,
    NormalizeStatement,
    SplitStatement,
    ModelStatement,
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
        if self.current_token().type == TokenType.IF:
            return self.parse_if_statement()

        elif self.current_token().type == TokenType.WHILE:
            return self.parse_while_statement()
        
        elif self.current_token().type == TokenType.PRINT or self.current_token().type == TokenType.PRINTLN:
            is_println = self.current_token().type == TokenType.PRINTLN
            self.eat(self.current_token().type)
            expr = self.parse_comparison()
            self.eat(TokenType.SEMICOLON)
            return PrintStatement(expression=expr, newline=is_println)
        
        elif self.current_token().type == TokenType.LOAD:
            return self.parse_load_statement()

        elif self.current_token().type == TokenType.DROP:
            return self.parse_drop_statement()

        elif self.current_token().type == TokenType.NORMALIZE:
            return self.parse_normalize_statement()

        elif self.current_token().type == TokenType.SPLIT:
            return self.parse_split_statement()

        elif self.current_token().type == TokenType.MODEL:
            return self.parse_model_statement()

        elif self.current_token().type == TokenType.TRAIN:
            return self.parse_train_statement()
        
        elif self.current_token().type == TokenType.IDENTIFIER and self.peek_token().type == TokenType.ASSIGN and self.peek_token(2).type == TokenType.EVALUATE:
            return self.parse_evaluate_statement()
        
        elif self.current_token().type == TokenType.IDENTIFIER:
            if self.peek_token().type == TokenType.ASSIGN:  # look ahead
                var_token = self.eat(TokenType.IDENTIFIER)
                self.eat(TokenType.ASSIGN)
                expr = self.parse_comparison()
                self.eat(TokenType.SEMICOLON)
                return AssignmentStatement(variable=Variable(name=var_token.value), expression=expr)
            else:
                expr = self.parse_comparison()  # treat as expression
                self.eat(TokenType.SEMICOLON)
                return expr

        else:
            expr = self.parse_comparison()
            self.eat(TokenType.SEMICOLON)
            return expr
        

    def parse_comparison(self) -> Any:
        """
        Handles comparison operators: ==, !=, <, >, <=, >=
        These have lower precedence than arithmetic but higher than assignment.

        comparison = expression ((==|!=|<|>|<=|>=) expression)*
        """
        node = self.parse_expression()

        comparison_ops = (
            TokenType.EQ, TokenType.NEQ,
            TokenType.LT, TokenType.GT,
            TokenType.LTE, TokenType.GTE,
        )

        while self.current_token().type in comparison_ops:
            op_token = self.eat(self.current_token().type)
            right = self.parse_expression()
            node = BinaryExpression(left=node, operator=op_token.value, right=right)

        return node

    def parse_block(self) -> List[Any]:
        """
        Parse a block of statements enclosed in braces.
        block = '{' statement* '}'
        Returns a list of AST nodes.
        """
        self.eat(TokenType.LBRACE)
        statements = []
        while self.current_token().type not in (TokenType.RBRACE, TokenType.EOF):
            statements.append(self.parse_statement())
        self.eat(TokenType.RBRACE)
        return statements

    def parse_if_statement(self) -> IfStatement:
        """
        Parse an if/else conditional statement.
        if_stmt = 'if' '(' comparison ')' block [ 'else' block ]
        """
        self.eat(TokenType.IF)
        self.eat(TokenType.LPAREN)
        condition = self.parse_comparison()
        self.eat(TokenType.RPAREN)
        then_branch = self.parse_block()

        else_branch = []
        if self.current_token().type == TokenType.ELSE:
            self.eat(TokenType.ELSE)
            else_branch = self.parse_block()

        return IfStatement(condition=condition, then_branch=then_branch, else_branch=else_branch)

    def parse_while_statement(self) -> WhileStatement:
        """
        Parse a while loop statement.
        while_stmt = 'while' '(' comparison ')' block
        """
        self.eat(TokenType.WHILE)
        self.eat(TokenType.LPAREN)
        condition = self.parse_comparison()
        self.eat(TokenType.RPAREN)
        body = self.parse_block()

        return WhileStatement(condition=condition, body=body)

    def parse_load_statement(self) -> LoadStatement:
        """load "path/to/file.csv";"""
        self.eat(TokenType.LOAD)
        path_token = self.eat(TokenType.STRING)
        self.eat(TokenType.SEMICOLON)
        return LoadStatement(filename=path_token.value)


    def parse_drop_statement(self) -> DropStatement:
        """drop columns "col1", "col2";"""
        self.eat(TokenType.DROP)
        self.eat(TokenType.COLUMNS)
        columns = self._parse_string_list()
        self.eat(TokenType.SEMICOLON)
        return DropStatement(column_names=columns)


    def parse_normalize_statement(self) -> NormalizeStatement:
        """normalize columns "col1", "col2";"""
        self.eat(TokenType.NORMALIZE)
        self.eat(TokenType.COLUMNS)
        columns = self._parse_string_list()
        self.eat(TokenType.SEMICOLON)
        return NormalizeStatement(column_names=columns)



    def _parse_string_list(self) -> List[str]:
        """
        Helper - parse one or more comma-separated string literals.
        Used by both drop and normalize.
        Returns a list of string values.
        """
        strings = []
        strings.append(self.eat(TokenType.STRING).value)
        while self.current_token().type == TokenType.COMMA:
            self.eat(TokenType.COMMA)
            strings.append(self.eat(TokenType.STRING).value)
        return strings


    def parse_split_statement(self) -> SplitStatement:
        """split data into train=0.8 test=0.2;"""
        self.eat(TokenType.SPLIT)
        self.eat(TokenType.DATA)
        self.eat(TokenType.INTO)

        # train=0.8 — TRAIN is a keyword, accepted here as param name
        self.eat(TokenType.TRAIN)
        self.eat(TokenType.ASSIGN)
        train_token = self.eat(TokenType.FLOAT)

        # test=0.2 — TEST is a keyword, accepted here as param name
        self.eat(TokenType.TEST)
        self.eat(TokenType.ASSIGN)
        test_token = self.eat(TokenType.FLOAT)

        self.eat(TokenType.SEMICOLON)
        return SplitStatement(
            train=train_token.value,
            test=test_token.value
        )


    def parse_model_statement(self) -> ModelStatement:
        """
        model RandomForest;
        model KNN neighbors=5;
        model Ridge alpha=0.1;
        """
        self.eat(TokenType.MODEL)
        model_name_token = self.eat(TokenType.IDENTIFIER)

        # collect optional key=value params before semicolon
        params = {}
        while self.current_token().type == TokenType.IDENTIFIER:
            key_token = self.eat(TokenType.IDENTIFIER)
            self.eat(TokenType.ASSIGN)
            val_token = self.parse_factor()   # handles NUMBER and FLOAT
            params[key_token.value] = val_token.value

        self.eat(TokenType.SEMICOLON)
        return ModelStatement(
            model_name=model_name_token.value,
            params=params
        )


    def parse_train_statement(self) -> TrainStatement:
        """train on train_set;"""
        self.eat(TokenType.TRAIN)
        self.eat(TokenType.ON)
        dataset_token = self.eat(TokenType.IDENTIFIER)
        self.eat(TokenType.SEMICOLON)
        return TrainStatement(dataset=dataset_token.value)


    def parse_evaluate_statement(self) -> EvaluateStatement:
        """accuracy = evaluate on test_set;"""
        result_var_token = self.eat(TokenType.IDENTIFIER)
        self.eat(TokenType.ASSIGN)
        self.eat(TokenType.EVALUATE)
        self.eat(TokenType.ON)
        dataset_token = self.eat(TokenType.IDENTIFIER)
        self.eat(TokenType.SEMICOLON)
        return EvaluateStatement(
            result_var=result_var_token.value,
            dataset=dataset_token.value
        )

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
        
        elif token.type == TokenType.FLOAT:
            self.eat(TokenType.FLOAT)
            return FloatLiteral(value=token.value)
        
        elif token.type == TokenType.STRING:
            self.eat(TokenType.STRING)
            return StringLiteral(value=token.value)
        
        elif token.type == TokenType.BOOLEAN:
            self.eat(TokenType.BOOLEAN)
            return BoolLiteral(value=token.value)
        
        elif token.type == TokenType.IDENTIFIER: # Treat identifiers as variables (could be part of an expression or a statement)
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
        
    def peek_token(self, n: int = 1) -> Token:
        if self.pos + n < len(self.tokens):
            return self.tokens[self.pos + n]
        return None

