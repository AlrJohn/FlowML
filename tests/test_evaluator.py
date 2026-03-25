"""Tests for the FlowML evaluator."""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flowml import interpret


def run(source: str):
    """Lex, parse, and evaluate source; return list of statement results."""
    from flowml.lexer import Lexer
    from flowml.parser import Parser
    from flowml.evaluator import Evaluator
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    return Evaluator().evaluate(ast)


def test_comparison_equal():
    """== returns True when operands are equal."""
    assert run("5 == 5;") == [True]

def test_comparison_not_equal():
    """!= returns True when operands differ."""
    assert run("5 != 3;") == [True]

def test_if_true_branch():
    """if executes then_block when condition is truthy."""
    result = run("x = 10; if (x > 5) { x = 99; } x;")
    assert result[-1] == 99

def test_if_false_branch():
    """else executes when condition is falsy."""
    result = run("x = 3; if (x > 5) { x = 99; } else { x = 0; } x;")
    assert result[-1] == 0

def test_if_no_else_skipped():
    """if with no else does nothing when condition is falsy."""
    result = run("x = 3; if (x > 5) { x = 99; } x;")
    assert result[-1] == 3

def test_while_counts():
    """while loop runs correct number of times."""
    result = run("i = 0; while (i < 5) { i = i + 1; } i;")
    assert result[-1] == 5

def test_while_never_executes():
    """while loop with false condition never runs body."""
    result = run("x = 10; while (x < 5) { x = 0; } x;")
    assert result[-1] == 10


