"""
Lexer (Tokenizer) for the FlowML Language
Converts source code text into tokens
"""

from .tokens import TokenType, Token


class Lexer:
    """Converts source code into tokens"""
    
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.current_char = self.source[0] if source else None
    
    def error(self, msg: str):
        """Raise a lexer error with position info"""
        raise Exception(f"Lexer Error at {self.line}:{self.column}: {msg}")
    
    def advance(self):
        """Move to the next character"""
        if self.current_char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        
        self.pos += 1
        if self.pos < len(self.source):
            self.current_char = self.source[self.pos]
        else:
            self.current_char = None
    
    def peek(self, offset: int = 1):
        """Look ahead without consuming"""
        peek_pos = self.pos + offset
        if peek_pos < len(self.source):
            return self.source[peek_pos]
        return None
    
    def skip_whitespace(self):
        """Skip whitespace and newlines"""
        while self.current_char and self.current_char.isspace():
            self.advance()
    
    def skip_comment(self):
        """Skip single-line comments starting with //"""
        if self.current_char == '/' and self.peek() == '/':
            while self.current_char and self.current_char != '\n':
                self.advance()
            self.advance()  # Skip the newline
    
    def read_number(self) -> Token:
        """Read a number literal"""
        start_line = self.line
        start_col = self.column
        num_str = ''
        
        while self.current_char and self.current_char.isdigit():
            num_str += self.current_char
            self.advance()
        
        # Check for float — decimal point followed by at least one digit
        if self.current_char == '.' and self.peek() and self.peek().isdigit():
            num_str += '.'
            self.advance()
            while self.current_char and self.current_char.isdigit():
                num_str += self.current_char
                self.advance()
            return Token(TokenType.FLOAT, float(num_str), start_line, start_col)

        # Integer fallback if no decimal point
        return Token(TokenType.NUMBER, int(num_str), start_line, start_col)
    
    def read_string(self) -> Token:
        """Read a double-quoted string literal, handling escape sequences."""
        start_line, start_col = self.line, self.column
        self.advance()  # skip opening "
        s = ''
        while self.current_char and self.current_char != '"':
            if self.current_char == '\\':  # Handle escape sequences
                self.advance()
                escapes = {'n': '\n', 't': '\t', '\\': '\\', '"': '"'}
                s += escapes.get(self.current_char, self.current_char)
            else:
                s += self.current_char
            self.advance()
        if not self.current_char:
            self.error("Unterminated string literal")
        self.advance()  # skip closing "
        return Token(TokenType.STRING, s, start_line, start_col)

    def read_identifier(self) -> Token:
        """Read an identifier or keyword"""
        start_line = self.line
        start_col = self.column
        id_str = ''
        
        while self.current_char and (self.current_char.isalnum() or self.current_char == '_'):
            id_str += self.current_char
            self.advance()
        
        # Check if it's a keyword
        keywords = {
            'print': TokenType.PRINT,
            'println': TokenType.PRINTLN,
            'if': TokenType.IF,
            'else': TokenType.ELSE,
            'while': TokenType.WHILE,
            'function': TokenType.FUNCTION,
            'return': TokenType.RETURN,
            'true': TokenType.BOOLEAN,
            'false': TokenType.BOOLEAN,
            'load': TokenType.LOAD,
            'drop': TokenType.DROP,
            'normalize': TokenType.NORMALIZE,
            'split': TokenType.SPLIT,
            'model': TokenType.MODEL,
            'train': TokenType.TRAIN,
            'test': TokenType.TEST,
            'target': TokenType.TARGET,
            'evaluate': TokenType.EVALUATE,
            'on': TokenType.ON,
            'into': TokenType.INTO,
            'column': TokenType.COLUMN, #Will most likely be removed in favor of COLUMNS
            'columns': TokenType.COLUMNS,
            'data': TokenType.DATA,
        }

        token_type = keywords.get(id_str, TokenType.IDENTIFIER)
        value = True if id_str == 'true' else (False if id_str == 'false' else id_str)
        return Token(token_type, value, start_line, start_col)
    
    def get_next_token(self) -> Token:
        """Get the next token from the source"""
        while self.current_char:
            # Skip whitespace
            if self.current_char.isspace():
                self.skip_whitespace()
                continue
            
            # Skip comments
            if self.current_char == '/' and self.peek() == '/':
                self.skip_comment()
                continue
            
            # Save position for token
            line = self.line
            col = self.column
            
            # Numbers
            if self.current_char.isdigit():
                return self.read_number()
            
            # Identifiers and keywords
            if self.current_char.isalpha() or self.current_char == '_':
                return self.read_identifier()
            
            # Two-character operators — must appear before single-character block
            if self.current_char == '=' and self.peek() == '=':
                self.advance(); self.advance()
                return Token(TokenType.EQ, '==', line, col)

            if self.current_char == '!' and self.peek() == '=':
                self.advance(); self.advance()
                return Token(TokenType.NEQ, '!=', line, col)

            if self.current_char == '<' and self.peek() == '=':
                self.advance(); self.advance()
                return Token(TokenType.LTE, '<=', line, col)

            if self.current_char == '>' and self.peek() == '=':
                self.advance(); self.advance()
                return Token(TokenType.GTE, '>=', line, col)

            if self.current_char == '=':
                self.advance()
                return Token(TokenType.ASSIGN, '=', line, col)

            # String literals
            if self.current_char == '"':
                return self.read_string()

            # Single-character tokens
            if self.current_char == '+':
                self.advance()
                return Token(TokenType.PLUS, '+', line, col)
            
            if self.current_char == '-':
                self.advance()
                return Token(TokenType.MINUS, '-', line, col)
            
            if self.current_char == '*':
                self.advance()
                return Token(TokenType.MULTIPLY, '*', line, col)
            
            if self.current_char == '/':
                self.advance()
                return Token(TokenType.DIVIDE, '/', line, col)
            
            if self.current_char == '(':
                self.advance()
                return Token(TokenType.LPAREN, '(', line, col)
            
            if self.current_char == ')':
                self.advance()
                return Token(TokenType.RPAREN, ')', line, col)
            
            if self.current_char == ';':
                self.advance()
                return Token(TokenType.SEMICOLON, ';', line, col)

            if self.current_char == '<':
                self.advance()
                return Token(TokenType.LT, '<', line, col)

            if self.current_char == '>':
                self.advance()
                return Token(TokenType.GT, '>', line, col)

            if self.current_char == '{':
                self.advance()
                return Token(TokenType.LBRACE, '{', line, col)

            if self.current_char == '}':
                self.advance()
                return Token(TokenType.RBRACE, '}', line, col)

            if self.current_char == ',':
                self.advance()
                return Token(TokenType.COMMA, ',', line, col)

            # Unknown character
            self.error(f"Unexpected character: '{self.current_char}'")
        
        # End of file
        return Token(TokenType.EOF, None, self.line, self.column)
    
    def tokenize(self):
        """Tokenize the entire source code"""
        tokens = []
        while True:
            token = self.get_next_token()
            tokens.append(token)
            if token.type == TokenType.EOF:
                break
        return tokens



