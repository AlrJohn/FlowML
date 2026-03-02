"""
Lexer (Tokenizer) for Toy Language
Converts source code text into tokens
"""

from dataclasses import dataclass
from typing import Any, List, Optional

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
    
    def peek(self, offset: int = 1) -> Optional[str]:
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
        
        return Token(TokenType.NUMBER, int(num_str), start_line, start_col)
    
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
        }
        
        token_type = keywords.get(id_str)
        if token_type:
            return Token(token_type, id_str, start_line, start_col)
        
        elif id_str: # It's an identifier
            return Token(TokenType.IDENTIFIER, id_str, start_line, start_col)
        
        else:
            self.error(f"Unknown identifier: {id_str}")
    
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
            
            if self.current_char == '=':
                self.advance()
                return Token(TokenType.ASSIGN, '=', line, col)

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
            
            # Unknown character
            self.error(f"Unexpected character: '{self.current_char}'")
        
        # End of file
        return Token(TokenType.EOF, None, self.line, self.column)
    
    def tokenize(self) -> List[Token]:
        """Tokenize the entire source code"""
        tokens = []
        while True:
            token = self.get_next_token()
            tokens.append(token)
            if token.type == TokenType.EOF:
                break
        return tokens



