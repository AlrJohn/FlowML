"""
FlowML ML Pipeline Tests
========================
Run with: python -m pytest tests/test_ml.py -v
Or:        python tests/test_ml.py

Assumes iris.csv is in the same directory as main.py.
"""

import sys
import os
import pytest

# Add the project root to path so we can import flowml
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flowml import interpret

# Path to iris.csv — adjust if needed
CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'iris.csv')


def run(source: str):
    """Run a FlowML program and return the results list."""
    return interpret(source)


def run_with_csv(source: str):
    """
    Run a FlowML program that references iris.csv.
    Temporarily changes working directory so relative path resolves.
    """
    original_dir = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(CSV_PATH)))
    try:
        return interpret(source)
    finally:
        os.chdir(original_dir)


def get_score(source: str) -> float:
    """
    Run a FlowML program and extract the final numeric result.
    Handles both a raw float and a list containing a float.
    """
    result = run_with_csv(source)
    if isinstance(result, list):
        # find the last numeric value in the results
        for val in reversed(result):
            if isinstance(val, (int, float)):
                return float(val)
    if isinstance(result, (int, float)):
        return float(result)
    raise ValueError(f"Could not extract score from result: {result}")


# =============================================================================
# HAPPY PATH TESTS
# =============================================================================

class TestMLHappyPath:

    def test_random_forest_full_pipeline(self):
        """Full pipeline with RandomForest should score between 0.85 and 1.0."""
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
        """LogisticRegression with no params should work and score above 0.7."""
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
        """KNN with neighbors=5 param should work correctly."""
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
        """SVM classifier should work and score above 0.8."""
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
        """NaiveBayes classifier should work and score above 0.7."""
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
        """Dropping one column should still allow training."""
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
        """Normalize before split — scaling should be deferred and applied correctly."""
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
        """Drop one column then normalize the remaining ones."""
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
        """evaluate result must be stored in self.variables so it can be used later."""
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
        """Ridge with alpha param should work — tests float param translation."""
        score = get_score("""
            load "iris.csv";
            split data into train=0.8 test=0.2;
            model Ridge alpha=0.1;
            train on train_set;
            score = evaluate on test_set;
            print score;
        """)
        # Ridge on iris (regression on categorical labels) — score may vary widely
        assert isinstance(score, float), f"Expected float score, got {score}"

    def test_random_forest_trees_param_translated(self):
        """trees=100 must be translated to n_estimators=100 for sklearn."""
        # If param translation fails sklearn raises TypeError — test would error
        score = get_score("""
            load "iris.csv";
            split data into train=0.8 test=0.2;
            model RandomForest trees=50;
            train on train_set;
            score = evaluate on test_set;
            print score;
        """)
        assert 0.85 <= score <= 1.0, f"Expected score in [0.85, 1.0], got {score}"


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestMLErrorHandling:

    def test_unknown_model_raises_error(self):
        """Unknown model name must raise a clear exception."""
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
        """train=0.8 test=0.3 sums to 1.1 — must raise an error."""
        with pytest.raises(Exception, match="(?i)sum|ratio|1.0|1\.1"):
            run_with_csv("""
                load "iris.csv";
                split data into train=0.8 test=0.3;
            """)

    def test_train_before_model_raises_error(self):
        """Calling train before model must raise a clear exception."""
        with pytest.raises(Exception, match="(?i)model|no model|undefined"):
            run_with_csv("""
                load "iris.csv";
                split data into train=0.8 test=0.2;
                train on train_set;
            """)

    def test_file_not_found_raises_error(self):
        """Loading a nonexistent file must raise an error."""
        with pytest.raises(Exception):
            run("""
                load "nonexistent_file_xyz.csv";
            """)

    def test_undefined_dataset_in_train_raises_error(self):
        """Referencing an undefined variable in train must raise an error."""
        with pytest.raises(Exception, match="(?i)undefined|not found|unknown"):
            run_with_csv("""
                load "iris.csv";
                model RandomForest;
                train on undefined_set;
            """)

    def test_undefined_dataset_in_evaluate_raises_error(self):
        """Referencing an undefined variable in evaluate must raise an error."""
        with pytest.raises(Exception, match="(?i)undefined|not found|unknown"):
            run_with_csv("""
                load "iris.csv";
                split data into train=0.8 test=0.2;
                model RandomForest;
                train on train_set;
                score = evaluate on undefined_set;
            """)


# =============================================================================
# NORMALIZATION CORRECTNESS TESTS
# =============================================================================

class TestNormalizationDeferred:

    def test_normalize_does_not_scale_immediately(self):
        """
        Normalization must be deferred until split.
        If normalize scales before split, the scaler would be fit on test data
        too (data leakage). We can't directly test internal state, but we can
        confirm the pipeline produces valid results when normalize comes before
        split, which requires correct deferred execution.
        """
        score = get_score("""
            load "iris.csv";
            normalize columns "sepal_length", "petal_length";
            split data into train=0.8 test=0.2;
            model KNN neighbors=5;
            train on train_set;
            score = evaluate on test_set;
            print score;
        """)
        assert 0.85 <= score <= 1.0, (
            f"Deferred normalization produced unexpected score: {score}"
        )

    def test_no_normalize_also_works(self):
        """Pipeline without normalize must also work — pending_normalize stays empty."""
        score = get_score("""
            load "iris.csv";
            split data into train=0.8 test=0.2;
            model RandomForest;
            train on train_set;
            score = evaluate on test_set;
            print score;
        """)
        assert 0.85 <= score <= 1.0, f"Pipeline without normalize failed: {score}"


# =============================================================================
# STANDALONE RUNNER
# =============================================================================

if __name__ == "__main__":
    """Run all tests manually without pytest."""
    import traceback

    tests = [
        # Happy path
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
        # Error handling
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
    print("  FlowML ML Pipeline Tests")
    print("=" * 60 + "\n")

    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  [PASS]  {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL]  {name}")
            print(f"          {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  {passed} passed, {failed} failed")
    print(f"{'=' * 60}\n")