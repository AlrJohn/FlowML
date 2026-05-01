# FlowML

FlowML is a domain-specific language (DSL) for machine learning workflows.
It lets you write small, readable `.fml` programs that run common ML steps
without writing raw pandas/scikit-learn code every time.

This repository now supports three execution modes:
1. Interpret and run a FlowML program (default mode).
2. Run semantic analysis only (`--analyze`).
3. Compile FlowML to standalone Python (`--compile`).

---

## Latest Additions (Current Version)

The current codebase includes these newer features:

1. Static semantic analyzer before execution/compilation.
2. HashTable-backed symbol table for variable and dataset type tracking.
3. Function definition/call support with `return`.
4. While-loops and comparison operators in the core language.
5. FlowML-to-Python code generation (`codegen.py`) used by `--compile`.
6. ML pipeline checks (load/split/model/train/evaluate ordering checks).
7. Optional `target="column_name"` in `split` statements.

---

## Prerequisites

- Python 3.8+
- `pip`

---

## Step 1 - Install

```bash
git clone https://github.com/AlrJohn/FlowML.git
cd FlowML
pip install -r requirements.txt
```

---

## Step 2 - Run Programs

### A) Default mode (analyze, then interpret)

```bash
python main.py your_program.fml
```

Behavior:
- FlowML first runs semantic checks.
- If there are semantic errors, it prints them and stops.
- If there are no semantic errors, it interprets the program.

### B) Semantic analysis only

```bash
python main.py your_program.fml --analyze
```

### C) Compile to Python

```bash
python main.py your_program.fml --compile
```

Expected compile output:
- `your_program.py` is generated next to the source file.
- If the source has semantic errors, compilation is aborted.

Note:
- `.toy` input files are also accepted in compile mode and are converted to `.py`.

---

## Example FlowML Programs

### Example 1 - Core language

```flowml
function add(a, b) {
    return a + b;
}

i = 0;
while (i < 3) {
    println add(i, 10);
    i = i + 1;
}
```

### Example 2 - ML pipeline

```flowml
load "iris.csv";
drop columns "sepal_width";
normalize columns "sepal_length", "petal_length", "petal_width";
split data into train=0.8 test=0.2 target="species";
model RandomForest trees=100;
train on train_set;
accuracy = evaluate on test_set;
println accuracy;
```

Supported model names:
- `LogisticRegression`
- `RandomForest`
- `SVM`
- `KNN`
- `NaiveBayes`
- `LinearRegression`
- `Ridge`
- `Lasso`
- `KMeans`

Supported parameter aliases:
- `trees` -> `n_estimators`
- `neighbors` -> `n_neighbors`
- `clusters` -> `n_clusters`
- `alpha` -> `alpha`

---

## Semantic Analyzer Checks

The analyzer currently validates:

1. Undefined variable use.
2. Type mismatches in strict assignment mode.
3. Invalid operand types in arithmetic/comparison expressions.
4. Unknown model names.
5. Split ratio correctness (`train + test == 1.0`).
6. Pipeline ordering errors:
   - `drop/normalize/split` before `load`
   - `train` before `model` or before `split`
   - `evaluate` before `train`

---

## Testing

Run the full test suite:

```bash
pytest tests/test_suite.py -v
```

Or run directly:

```bash
python tests/test_suite.py
```

The test suite covers:
- Core evaluator behavior (conditionals, loops, comparisons).
- ML pipeline happy paths.
- Error handling cases.
- Deferred normalization behavior.

---

## Project Structure

```text
FlowML/
|-- main.py
|-- requirements.txt
|-- README.md
|-- iris.csv
|-- titanic.csv
|-- flowml/
|   |-- __init__.py
|   |-- tokens.py
|   |-- lexer.py
|   |-- parser.py
|   |-- ast_nodes.py
|   |-- evaluator.py
|   |-- environment.py
|   |-- semantic_analyzer.py
|   |-- symbol_table.py
|   |-- MLBackend.py
|   `-- codegen.py
|-- tests/
|   |-- __init__.py
|   `-- test_suite.py

```

---

## Developer Notes

The internal API exposed in `flowml/__init__.py`:

- `interpret(source: str)`
- `analyze(source: str, strict_types: bool = True)`
- `compile_to_python(source: str) -> str`

This keeps CLI behavior and programmatic usage aligned.
