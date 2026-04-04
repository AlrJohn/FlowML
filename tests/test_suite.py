"""
FlowML Test Suite
=================
Covers:
  - Core language (evaluator): comparisons, conditionals, while loops
  - ML pipeline: model training, normalization, error handling

Run with:  pytest tests/test_suite.py -v
       or:  python tests/test_suite.py
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flowml import interpret
from flowml.lexer import Lexer
from flowml.parser import Parser
from flowml.evaluator import Evaluator

CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'iris.csv')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(source: str):
    """Lex, parse, and evaluate source; return list of statement results."""
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    return Evaluator().evaluate(ast)


def run_with_csv(source: str):
    """Run a FlowML program that references iris.csv."""
    original_dir = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(CSV_PATH)))
    try:
        return interpret(source)
    finally:
        os.chdir(original_dir)


def get_score(source: str) -> float:
    """Run a FlowML program and extract the final numeric result."""
    result = run_with_csv(source)
    if isinstance(result, list):
        for val in reversed(result):
            if isinstance(val, (int, float)):
                return float(val)
    if isinstance(result, (int, float)):
        return float(result)
    raise ValueError(f"Could not extract score from result: {result}")


# ===========================================================================
# CORE LANGUAGE TESTS
# ===========================================================================

def test_comparison_equal():
    assert run("5 == 5;") == [True]

def test_comparison_not_equal():
    assert run("5 != 3;") == [True]

def test_if_true_branch():
    result = run("x = 10; if (x > 5) { x = 99; } x;")
    assert result[-1] == 99

def test_if_false_branch():
    result = run("x = 3; if (x > 5) { x = 99; } else { x = 0; } x;")
    assert result[-1] == 0

def test_if_no_else_skipped():
    result = run("x = 3; if (x > 5) { x = 99; } x;")
    assert result[-1] == 3

def test_while_counts():
    result = run("i = 0; while (i < 5) { i = i + 1; } i;")
    assert result[-1] == 5

def test_while_never_executes():
    result = run("x = 10; while (x < 5) { x = 0; } x;")
    assert result[-1] == 10


# ===========================================================================
# ML PIPELINE — HAPPY PATH
# ===========================================================================

class TestMLHappyPath:

    def test_random_forest_full_pipeline(self):
        score = get_score("""
            load "iris.csv";
            normalize columns "sepal_length", "sepal_width", "petal_length", "petal_width";
            split data into train=0.8 test=0.2;
            model RandomForest trees=100;
            train on train_set;
            accuracy = evaluate on test_set;
            print accuracy;
        """)
        assert 0.85 <= score <= 1.0, f"Expected score in [0.85, 1.0], got {score}"

    def test_logistic_regression_no_params(self):
        score = get_score("""
            load "iris.csv";
            split data into train=0.8 test=0.2;
            model LogisticRegression;
            train on train_set;
            score = evaluate on test_set;
            print score;
        """)
        assert 0.7 <= score <= 1.0, f"Expected score in [0.7, 1.0], got {score}"

    def test_knn_with_neighbors_param(self):
        score = get_score("""
            load "iris.csv";
            split data into train=0.8 test=0.2;
            model KNN neighbors=5;
            train on train_set;
            score = evaluate on test_set;
            print score;
        """)
        assert 0.85 <= score <= 1.0, f"Expected score in [0.85, 1.0], got {score}"

    def test_svm(self):
        score = get_score("""
            load "iris.csv";
            split data into train=0.8 test=0.2;
            model SVM;
            train on train_set;
            score = evaluate on test_set;
            print score;
        """)
        assert 0.8 <= score <= 1.0, f"Expected score in [0.8, 1.0], got {score}"

    def test_naive_bayes(self):
        score = get_score("""
            load "iris.csv";
            split data into train=0.8 test=0.2;
            model NaiveBayes;
            train on train_set;
            score = evaluate on test_set;
            print score;
        """)
        assert 0.7 <= score <= 1.0, f"Expected score in [0.7, 1.0], got {score}"

    def test_drop_columns_single(self):
        score = get_score("""
            load "iris.csv";
            drop columns "sepal_width";
            split data into train=0.8 test=0.2;
            model LogisticRegression;
            train on train_set;
            score = evaluate on test_set;
            print score;
        """)
        assert 0.6 <= score <= 1.0, f"Expected score in [0.6, 1.0], got {score}"

    def test_normalize_then_split(self):
        score = get_score("""
            load "iris.csv";
            normalize columns "sepal_length", "sepal_width", "petal_length", "petal_width";
            split data into train=0.8 test=0.2;
            model KNN neighbors=3;
            train on train_set;
            score = evaluate on test_set;
            print score;
        """)
        assert 0.85 <= score <= 1.0, f"Expected score in [0.85, 1.0], got {score}"

    def test_drop_and_normalize_together(self):
        score = get_score("""
            load "iris.csv";
            drop columns "sepal_width";
            normalize columns "sepal_length", "petal_length", "petal_width";
            split data into train=0.8 test=0.2;
            model RandomForest;
            train on train_set;
            score = evaluate on test_set;
            print score;
        """)
        assert 0.85 <= score <= 1.0, f"Expected score in [0.85, 1.0], got {score}"

    def test_score_stored_as_variable(self):
        score = get_score("""
            load "iris.csv";
            split data into train=0.8 test=0.2;
            model RandomForest;
            train on train_set;
            accuracy = evaluate on test_set;
            copy = accuracy;
            print copy;
        """)
        assert 0.85 <= score <= 1.0, f"Score variable not correctly stored: {score}"

    def test_ridge_regression_alpha_param(self):
        score = get_score("""
            load "iris.csv";
            split data into train=0.8 test=0.2;
            model Ridge alpha=0.1;
            train on train_set;
            score = evaluate on test_set;
            print score;
        """)
        assert isinstance(score, float), f"Expected float score, got {score}"

    def test_random_forest_trees_param_translated(self):
        score = get_score("""
            load "iris.csv";
            split data into train=0.8 test=0.2;
            model RandomForest trees=50;
            train on train_set;
            score = evaluate on test_set;
            print score;
        """)
        assert 0.85 <= score <= 1.0, f"Expected score in [0.85, 1.0], got {score}"


# ===========================================================================
# ML PIPELINE — ERROR HANDLING
# ===========================================================================

class TestMLErrorHandling:

    def test_unknown_model_raises_error(self):
        with pytest.raises(Exception, match="(?i)unknown|not found|invalid|FakeModel"):
            run_with_csv("""
                load "iris.csv";
                split data into train=0.8 test=0.2;
                model FakeModel;
                train on train_set;
                score = evaluate on test_set;
                print score;
            """)

    def test_split_ratios_not_summing_to_one(self):
        with pytest.raises(Exception, match="(?i)sum|ratio|1.0|1\.1"):
            run_with_csv("""
                load "iris.csv";
                split data into train=0.8 test=0.3;
            """)

    def test_train_before_model_raises_error(self):
        with pytest.raises(Exception, match="(?i)model|no model|undefined"):
            run_with_csv("""
                load "iris.csv";
                split data into train=0.8 test=0.2;
                train on train_set;
            """)

    def test_file_not_found_raises_error(self):
        with pytest.raises(Exception):
            run('load "nonexistent_file_xyz.csv";')

    def test_undefined_dataset_in_train_raises_error(self):
        with pytest.raises(Exception, match="(?i)undefined|not found|unknown"):
            run_with_csv("""
                load "iris.csv";
                model RandomForest;
                train on undefined_set;
            """)

    def test_undefined_dataset_in_evaluate_raises_error(self):
        with pytest.raises(Exception, match="(?i)undefined|not found|unknown"):
            run_with_csv("""
                load "iris.csv";
                split data into train=0.8 test=0.2;
                model RandomForest;
                train on train_set;
                score = evaluate on undefined_set;
            """)


# ===========================================================================
# ML PIPELINE — NORMALIZATION
# ===========================================================================

class TestNormalizationDeferred:

    def test_normalize_does_not_scale_immediately(self):
        score = get_score("""
            load "iris.csv";
            normalize columns "sepal_length", "petal_length";
            split data into train=0.8 test=0.2;
            model KNN neighbors=5;
            train on train_set;
            score = evaluate on test_set;
            print score;
        """)
        assert 0.85 <= score <= 1.0, f"Deferred normalization produced unexpected score: {score}"

    def test_no_normalize_also_works(self):
        score = get_score("""
            load "iris.csv";
            split data into train=0.8 test=0.2;
            model RandomForest;
            train on train_set;
            score = evaluate on test_set;
            print score;
        """)
        assert 0.85 <= score <= 1.0, f"Pipeline without normalize failed: {score}"


# ===========================================================================
# STANDALONE RUNNER
# ===========================================================================

if __name__ == "__main__":
    import traceback

    suites = [
        # Core language
        ("Comparison equal",            lambda: test_comparison_equal()),
        ("Comparison not equal",        lambda: test_comparison_not_equal()),
        ("If true branch",              lambda: test_if_true_branch()),
        ("If false branch",             lambda: test_if_false_branch()),
        ("If no else skipped",          lambda: test_if_no_else_skipped()),
        ("While counts",                lambda: test_while_counts()),
        ("While never executes",        lambda: test_while_never_executes()),
        # ML happy path
        ("RandomForest full pipeline",          lambda: TestMLHappyPath().test_random_forest_full_pipeline()),
        ("LogisticRegression no params",        lambda: TestMLHappyPath().test_logistic_regression_no_params()),
        ("KNN with neighbors param",            lambda: TestMLHappyPath().test_knn_with_neighbors_param()),
        ("SVM",                                 lambda: TestMLHappyPath().test_svm()),
        ("NaiveBayes",                          lambda: TestMLHappyPath().test_naive_bayes()),
        ("Drop single column",                  lambda: TestMLHappyPath().test_drop_columns_single()),
        ("Normalize then split",                lambda: TestMLHappyPath().test_normalize_then_split()),
        ("Drop and normalize together",         lambda: TestMLHappyPath().test_drop_and_normalize_together()),
        ("Score stored as variable",            lambda: TestMLHappyPath().test_score_stored_as_variable()),
        ("Ridge alpha param",                   lambda: TestMLHappyPath().test_ridge_regression_alpha_param()),
        ("RandomForest trees param translated", lambda: TestMLHappyPath().test_random_forest_trees_param_translated()),
        # ML error handling
        ("Unknown model raises error",          lambda: TestMLErrorHandling().test_unknown_model_raises_error()),
        ("Split ratios not summing to 1",       lambda: TestMLErrorHandling().test_split_ratios_not_summing_to_one()),
        ("Train before model raises error",     lambda: TestMLErrorHandling().test_train_before_model_raises_error()),
        ("File not found raises error",         lambda: TestMLErrorHandling().test_file_not_found_raises_error()),
        ("Undefined dataset in train",          lambda: TestMLErrorHandling().test_undefined_dataset_in_train_raises_error()),
        ("Undefined dataset in evaluate",       lambda: TestMLErrorHandling().test_undefined_dataset_in_evaluate_raises_error()),
        # Normalization
        ("Normalize deferred correctly",        lambda: TestNormalizationDeferred().test_normalize_does_not_scale_immediately()),
        ("No normalize also works",             lambda: TestNormalizationDeferred().test_no_normalize_also_works()),
    ]

    passed = 0
    failed = 0

    print("\n" + "=" * 60)
    print("  FlowML Test Suite")
    print("=" * 60 + "\n")

    for name, test_fn in suites:
        try:
            test_fn()
            print(f"  [PASS]  {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL]  {name}")
            print(f"          {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  {passed} passed, {failed} failed")
    print(f"{'=' * 60}\n")
