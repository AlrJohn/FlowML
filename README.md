# FlowML

FlowML is a domain-specific language (DSL) for building machine learning pipelines. It provides a clean, readable syntax that abstracts away scikit-learn, pandas, and numpy, letting you express a complete ML workflow in a handful of lines.

Programs use the `.fml` extension. FlowML is a tree-walking interpreter written in Python.

---

## Prerequisites

- Python 3.8 or later
- pip

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/AlrJohn/FlowML.git
cd FlowML

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Running a Program

```bash
python main.py your_program.fml
```

Run the semantic analysis pass only (no execution):

```bash
python main.py your_program.fml --analyze
```

**Example program** (`example.fml`):

```
load "iris.csv";
normalize columns "sepal_length", "sepal_width", "petal_length", "petal_width";
split data into train=0.8 test=0.2;
model RandomForest trees=100;
train on train_set;
accuracy = evaluate on test_set;
println accuracy;
```

---

## Running the Test Suite

```bash
# With pytest (recommended)
pytest tests/test_suite.py -v

# Without pytest
python tests/test_suite.py
```

---

## Project Structure

```
FlowML/
│
├── main.py                  # Entry point — run .fml programs from the command line
├── iris.csv                 # Sample dataset used by the ML pipeline tests
├── requirements.txt         # Python dependencies
│
├── flowml/                  # Interpreter source
│   ├── __init__.py          # interpret() and analyze() entry points
│   ├── tokens.py            # TokenType enum and Token dataclass
│   ├── lexer.py             # Source text → token list
│   ├── ast_nodes.py         # AST node dataclasses
│   ├── parser.py            # Token list → AST (recursive-descent parser)
│   ├── evaluator.py         # AST → execution (tree-walking interpreter)
│   ├── environment.py       # Lexical scope / variable lookup
│   ├── symbol_table.py      # HashTable-backed symbol table for type tracking
│   ├── semantic_analyzer.py # Static analysis: type checks, pipeline ordering
│   └── MLBackend.py         # scikit-learn / pandas integration
│
├── tests/
│   └── test_suite.py        # Full test suite (core language + ML pipeline)
│
└── docs/
    ├── flowml_language_reference.md    # Full language specification
    └── flowml_implementation_guide.md  # Architecture and design details
    ...
```

---

## Language Quick Reference

### Variables & Arithmetic

```
x = 10;
result = (x + 5) * 2;
```

### Conditionals

```
if (result > 20) {
    println "big";
} else {
    println "small";
}
```

### Loops

```
i = 0;
while (i < 5) {
    println i;
    i = i + 1;
}
```

### Functions

```
function add(a, b) {
    return a + b;
}
println add(3, 7);
```

### ML Pipeline

```
load "data.csv";
drop columns "irrelevant_col";
normalize columns "col1", "col2";
split data into train=0.8 test=0.2;
model RandomForest trees=100;
train on train_set;
accuracy = evaluate on test_set;
println accuracy;
```

**Supported models:** `LogisticRegression`, `RandomForest`, `SVM`, `KNN`, `NaiveBayes`, `LinearRegression`, `Ridge`, `Lasso`, `KMeans`

For the full language specification see [`docs/flowml_language_reference.md`](docs/flowml_language_reference.md).
