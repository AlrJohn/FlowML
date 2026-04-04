# FlowML DSL: Implementation Guide

This document explains how each component of the FlowML interpreter is implemented, with code-level detail.

---

## Architecture Overview

FlowML is a tree-walking interpreter built in Python. The architecture follows the classic interpreter pipeline:

```
Source Code Text
      ↓
   Lexer (lexer.py)
      ↓  tokenize()
Token Stream [Token, Token, Token, ...]
      ↓
   Parser (parser.py)
      ↓  parse()
Abstract Syntax Tree (AST)
      ├──► Semantic Analyzer (semantic_analyzer.py)  [optional static pass]
      │         uses SymbolTable (symbol_table.py)
      ↓
   Evaluator (evaluator.py)
      ↓  evaluate()
      │    uses Environment (environment.py)
Results / Side Effects
      ↓
MLBackend (MLBackend.py)
      ↓  sklearn operations
ML Output (accuracy, trained model, etc.)
```

**File structure:**
```
flowml/
├── __init__.py          # Package entry: interpret() and analyze() functions
├── tokens.py            # TokenType enum and Token dataclass
├── lexer.py             # Lexer: source text → token list
├── ast_nodes.py         # AST node dataclasses
├── parser.py            # Parser: token list → AST
├── evaluator.py         # Evaluator: AST → execution
├── environment.py       # Lexical scope management for the evaluator
├── symbol_table.py      # HashTable + SymbolTable for the semantic analyzer
├── semantic_analyzer.py # Static analysis pass (type checking, pipeline order)
└── MLBackend.py         # scikit-learn integration layer
```

**Entry functions (`__init__.py`):**
```python
def interpret(source: str) -> int:
    lexer  = Lexer(source)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    asts   = parser.parse()
    return Evaluator().evaluate(asts)
```

---

## Component 1: Token System (`tokens.py`)

### Design

Tokens are the atomic units of the language. Each token has:
- A **type** (from the `TokenType` enum)
- A **value** (the actual text or parsed literal value)
- A **line** and **column** for error reporting

```python
@dataclass
class Token:
    type: TokenType
    value: Any
    line: int
    column: int
```

### Token Types

The `TokenType` enum uses `auto()` for automatic integer assignment. All possible token types are:

**Literals:**
- `NUMBER` — integer like `5`, `42`
- `FLOAT` — decimal like `3.14`, `0.8`
- `STRING` — quoted text like `"iris.csv"`
- `BOOLEAN` — `true` or `false`

**Identifiers and Assignment:**
- `IDENTIFIER` — variable/model names like `x`, `accuracy`, `RandomForest`
- `ASSIGN` — the `=` symbol

**Arithmetic Operators:**
- `PLUS`, `MINUS`, `MULTIPLY`, `DIVIDE`

**Comparison Operators:**
- `EQ` (`==`), `NEQ` (`!=`), `LT` (`<`), `GT` (`>`), `LTE` (`<=`), `GTE` (`>=`)

**Control Flow Keywords:**
- `IF`, `ELSE`, `WHILE`, `FUNCTION`, `RETURN`

**ML Keywords:**
- `LOAD`, `DROP`, `NORMALIZE`, `SPLIT`, `MODEL`, `TRAIN`, `TEST`, `EVALUATE`
- `ON`, `INTO`, `COLUMNS`, `COLUMN`, `DATA`

**Punctuation:**
- `LPAREN` (`(`), `RPAREN` (`)`), `LBRACE` (`{`), `RBRACE` (`}`)
- `SEMICOLON` (`;`), `COMMA` (`,`)

**Special:**
- `EOF` — marks end of input

---

## Component 2: Lexer (`lexer.py`)

### Design

The `Lexer` class converts raw source code text into a flat list of tokens. It uses a single-character lookahead model with a `peek()` method for two-character operators.

### Key State

```python
self.source       # Full source string
self.pos          # Current position (index into source)
self.line         # Current line number (for error messages)
self.column       # Current column number (for error messages)
self.current_char # Character at self.pos
```

### Core Methods

**`advance()`** — Moves to the next character. Tracks newlines for line/column counting.
```python
def advance(self):
    if self.current_char == '\n':
        self.line += 1
        self.column = 1
    else:
        self.column += 1
    self.pos += 1
    self.current_char = self.source[self.pos] if self.pos < len(self.source) else None
```

**`peek(offset=1)`** — Returns the character `offset` positions ahead without advancing.

**`skip_whitespace()`** — Advances past all whitespace characters.

**`skip_comment()`** — Detects `//` and advances to end of line.

**`read_number()`** — Reads an integer, then checks if followed by `.digit` to produce a `FLOAT` token; otherwise produces `NUMBER`.

**`read_string()`** — Reads a double-quoted string, handling escape sequences `\n`, `\t`, `\\`, `\"`.

**`read_identifier()`** — Reads a word (letters/digits/underscores), then checks a keyword dictionary. Returns the appropriate keyword token type, or `IDENTIFIER` if not a keyword. Boolean literals `true` and `false` are converted to Python `True`/`False` here.

### Critical Implementation Detail: Two-Character Operators

Two-character operators (`==`, `!=`, `<=`, `>=`) are checked **before** single-character ones in `get_next_token()`. This prevents `==` from being tokenized as two separate `=` tokens:

```python
if self.current_char == '=' and self.peek() == '=':
    self.advance(); self.advance()
    return Token(TokenType.EQ, '==', line, col)

if self.current_char == '=':   # only reached if next char is NOT '='
    self.advance()
    return Token(TokenType.ASSIGN, '=', line, col)
```

### Tokenize Flow

`tokenize()` calls `get_next_token()` in a loop until an `EOF` token is produced, collecting all tokens into a list.

---

## Component 3: AST Nodes (`ast_nodes.py`)

All AST nodes are Python `@dataclass` objects. Dataclasses provide automatic `__init__`, `__repr__`, and `__eq__` methods.

### Expression Nodes

```python
@dataclass class NumberLiteral:   value: int
@dataclass class FloatLiteral:    value: float
@dataclass class StringLiteral:   value: str
@dataclass class BoolLiteral:     value: bool
@dataclass class Variable:        name: str
@dataclass class UnaryExpression: operator: str; operand: Any
@dataclass class BinaryExpression: left: Any; operator: str; right: Any
```

### Statement Nodes

```python
@dataclass class AssignmentStatement: variable: Variable; expression: Any
@dataclass class PrintStatement:      expression: Any; newline: bool = False
@dataclass class IfStatement:         condition: Any; then_branch: List[Any]; else_branch: List[Any]
@dataclass class WhileStatement:      condition: Any; body: List[Any]
```

### ML Statement Nodes

```python
@dataclass class LoadStatement:      filename: str
@dataclass class DropStatement:      column_names: List[str]
@dataclass class NormalizeStatement: column_names: List[str]
@dataclass class SplitStatement:     train: float; test: float
@dataclass class ModelStatement:     model_name: str; params: Optional[dict] = None
@dataclass class TrainStatement:     dataset: str
@dataclass class EvaluateStatement:  result_var: str; dataset: str
```

### Root Node

```python
@dataclass class Program: statements: List[Any]
```

---

## Component 4: Parser (`parser.py`)

### Design

The `Parser` implements a **recursive descent parser** — a collection of mutually recursive methods, one per grammar rule. Token consumption is managed by the `eat()` method.

### Key State

```python
self.tokens  # List of tokens from lexer
self.pos     # Current position in token list
```

### `eat(token_type)` — Token Consumption

The cornerstone of the parser. Checks that the current token matches the expected type; if so, consumes it and returns it. If not, raises a `Parser Error` with line/column information.

```python
def eat(self, token_type):
    token = self.current_token()
    if token and token.type == token_type:
        self.pos += 1
        return token
    raise Exception(f"Parser Error at {token.line}:{token.column}: Expected {token_type.name} but got {token.type.name}")
```

### `peek_token(n=1)` — Look-Ahead

Returns the token `n` positions ahead without consuming. Used in `parse_statement()` to distinguish between:
- `x = expr;` → AssignmentStatement
- `x = evaluate on y;` → EvaluateStatement (special case detected by 2-token lookahead)

```python
elif self.current_token().type == TokenType.IDENTIFIER and \
     self.peek_token().type == TokenType.ASSIGN and \
     self.peek_token(2).type == TokenType.EVALUATE:
    return self.parse_evaluate_statement()
```

### Expression Parsing (Precedence Climbing)

Operator precedence is encoded structurally through method call chains:

```
parse_comparison()       ← lowest precedence: ==, !=, <, >, <=, >=
    └── parse_expression()    ← + and -
            └── parse_term()        ← * and /
                    └── parse_unary()      ← unary -
                            └── parse_factor()     ← literals, variables, ( expr )
```

Each level calls the next higher level for operands. This guarantees that `*` binds tighter than `+`, which binds tighter than `==`.

### ML Statement Parsers

Each ML statement has a dedicated parser method that consumes the expected tokens in sequence:

**`parse_load_statement()`:**
```python
eat(LOAD) → eat(STRING) → eat(SEMICOLON)
→ LoadStatement(filename)
```

**`parse_drop_statement()`:**
```python
eat(DROP) → eat(COLUMNS) → _parse_string_list() → eat(SEMICOLON)
→ DropStatement(column_names)
```

**`parse_normalize_statement()`:**
```python
eat(NORMALIZE) → eat(COLUMNS) → _parse_string_list() → eat(SEMICOLON)
→ NormalizeStatement(column_names)
```

**`parse_split_statement()`:**
```python
eat(SPLIT) → eat(DATA) → eat(INTO) →
eat(TRAIN) → eat(ASSIGN) → eat(FLOAT) →
eat(TEST) → eat(ASSIGN) → eat(FLOAT) → eat(SEMICOLON)
→ SplitStatement(train, test)
```
Note: `TRAIN` and `TEST` are keywords repurposed as parameter names here.

**`parse_model_statement()`:**
```python
eat(MODEL) → eat(IDENTIFIER) →
  [while IDENTIFIER: eat(IDENTIFIER) eat(ASSIGN) parse_factor() → add to params dict]
→ eat(SEMICOLON)
→ ModelStatement(model_name, params)
```
Uses `parse_factor()` for parameter values to accept both NUMBER and FLOAT.

**`parse_train_statement()`:**
```python
eat(TRAIN) → eat(ON) → eat(IDENTIFIER) → eat(SEMICOLON)
→ TrainStatement(dataset)
```

**`parse_evaluate_statement()`:**
```python
eat(IDENTIFIER) → eat(ASSIGN) → eat(EVALUATE) → eat(ON) → eat(IDENTIFIER) → eat(SEMICOLON)
→ EvaluateStatement(result_var, dataset)
```

**`_parse_string_list()`:** Helper used by `drop` and `normalize`. Parses `STRING (',' STRING)*`.

---

## Component 5: Evaluator (`evaluator.py`)

### Design

The `Evaluator` implements a **tree-walking interpreter** using Python's `getattr` for dynamic dispatch. Each AST node type maps to a corresponding `eval_<NodeType>()` method.

### Key State

```python
self.variables   # dict: variable name → value (all runtime state here)
self.ml_backend  # MLBackend instance
```

### Dispatch Mechanism

```python
def evaluate(self, node):
    method_name = f"eval_{type(node).__name__}"
    method = getattr(self, method_name, self.no_eval)
    return method(node)
```

This uses Python reflection to automatically route `NumberLiteral` to `eval_NumberLiteral`, `WhileStatement` to `eval_WhileStatement`, etc. If no method exists, `no_eval()` raises an informative error.

### eval_BinaryExpression

Handles both arithmetic and comparison operators:

```python
def eval_BinaryExpression(self, node):
    left  = self.evaluate(node.left)
    right = self.evaluate(node.right)
    if node.operator == '+':  return left + right
    if node.operator == '-':  return left - right
    if node.operator == '*':  return left * right
    if node.operator == '/':
        if right == 0: raise Exception("Division by zero")
        return left // right   # integer division
    if node.operator == '==': return left == right
    if node.operator == '!=': return left != right
    if node.operator == '<':  return left < right
    if node.operator == '>':  return left > right
    if node.operator == '<=': return left <= right
    if node.operator == '>=': return left >= right
```

### eval_IfStatement

```python
def eval_IfStatement(self, node):
    condition = self.evaluate(node.condition)
    if condition:
        result = None
        for stmt in node.then_branch:
            result = self.evaluate(stmt)
        return result
    elif node.else_branch:
        result = None
        for stmt in node.else_branch:
            result = self.evaluate(stmt)
        return result
    return None
```

### eval_WhileStatement

```python
def eval_WhileStatement(self, node):
    result = None
    while self.evaluate(node.condition):
        for stmt in node.body:
            result = self.evaluate(stmt)
    return result
```

The condition is re-evaluated at the top of every iteration by calling `self.evaluate(node.condition)`.

### eval_SplitStatement

The split statement does double duty: it calls the MLBackend, then binds the results into the variable store:

```python
def eval_SplitStatement(self, node):
    self.ml_backend.split(node.train, node.test)
    self.variables['train_set'] = self.ml_backend.train_set
    self.variables['test_set']  = self.ml_backend.test_set
```

### eval_EvaluateStatement

Evaluates the model and immediately stores the score in a variable:

```python
def eval_EvaluateStatement(self, node):
    if node.dataset not in self.variables:
        raise Exception(f"Undefined variable '{node.dataset}'")
    dataset = self.variables[node.dataset]
    result  = self.ml_backend.evaluate(dataset)
    self.variables[node.result_var] = result
    return result
```

---

## Component 6: ML Backend (`MLBackend.py`)

### Design

The `MLBackend` class encapsulates all interaction with scikit-learn and pandas. It maintains the active DataFrame, the current model, and the train/test split results.

### Key State

```python
self.df                   # Active pandas DataFrame (after load)
self.train_set            # (X_train, y_train) tuple after split
self.test_set             # (X_test, y_test) tuple after split
self.model                # Trained sklearn model instance
self.normalization_columns # List of column names for deferred scaling
self._target_encoding_map  # For encoding categorical targets in regression
```

### MODEL_MAP and PARAM_MAP

Two module-level dictionaries power the model instantiation system:

```python
MODEL_MAP = {
    'LogisticRegression': LogisticRegression,
    'RandomForest':       RandomForestClassifier,
    'SVM':                SVC,
    'KNN':                KNeighborsClassifier,
    'NaiveBayes':         GaussianNB,
    'LinearRegression':   LinearRegression,
    'Ridge':              Ridge,
    'Lasso':              Lasso,
    'KMeans':             KMeans,
}

PARAM_MAP = {
    'trees':     'n_estimators',
    'neighbors': 'n_neighbors',
    'clusters':  'n_clusters',
    'alpha':     'alpha',
}
```

### set_model()

```python
def set_model(self, model_name, params=None):
    if model_name not in MODEL_MAP:
        raise Exception(f"MLBackend: Unknown model type '{model_name}'")
    sklearn_params = {}
    for key, val in (params or {}).items():
        sklearn_key = PARAM_MAP.get(key, key)  # translate or pass through
        sklearn_params[sklearn_key] = val
    self.model = MODEL_MAP[model_name](**sklearn_params)
    self._target_encoding_map = None
```

### split() — The Deferred Normalization Pattern

The most architecturally interesting method. Normalization is applied here, after splitting, to prevent fitting the scaler on test data:

```python
def split(self, train, test):
    if not (train + test == 1.0):
        raise Exception("Train and test ratios must sum to 1")

    X = self.df.iloc[:, :-1]   # all columns except last = features
    y = self.df.iloc[:, -1]    # last column = target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test)

    if self.normalization_columns:
        scaler = StandardScaler()
        X_train[self.normalization_columns] = scaler.fit_transform(X_train[self.normalization_columns])
        X_test[self.normalization_columns]  = scaler.transform(X_test[self.normalization_columns])
        self.normalization_columns = []   # reset after applying

    self.train_set = (X_train, y_train)
    self.test_set  = (X_test, y_test)
```

Key: `fit_transform` on training data only; `transform` (not `fit_transform`) on test data.

### train() — Categorical Target Encoding

Regression models require numeric targets. If the target column contains strings (e.g., species names in iris), they are encoded as integers:

```python
def train(self, dataset):
    if self.model is None:
        raise Exception("No model defined. Call 'model' before 'train'.")
    X, y = dataset
    if is_regressor(self.model) and not pd.api.types.is_numeric_dtype(y):
        unique_labels = pd.unique(y)
        self._target_encoding_map = {label: idx for idx, label in enumerate(unique_labels)}
        y = y.map(self._target_encoding_map).astype(float)
    self.model.fit(X, y)
```

The same encoding map is stored and reused in `evaluate()` for consistent scoring.

### evaluate()

```python
def evaluate(self, dataset):
    if self.model is None:
        raise Exception("No model defined.")
    X, y = dataset
    if self._target_encoding_map is not None:
        y = y.map(self._target_encoding_map).fillna(-1).astype(float)
    return self.model.score(X, y)
```

---

## Data Flow Trace: Full Pipeline Execution

For this program:
```
load "iris.csv";
normalize columns "sepal_length";
split data into train=0.8 test=0.2;
model RandomForest trees=100;
train on train_set;
accuracy = evaluate on test_set;
println accuracy;
```

**Step 1 — Lexer:** Converts the 7 statements into a stream of ~60 tokens including LOAD, STRING("iris.csv"), SEMICOLON, NORMALIZE, COLUMNS, STRING("sepal_length"), SEMICOLON, SPLIT, DATA, INTO, TRAIN, ASSIGN, FLOAT(0.8), TEST, ASSIGN, FLOAT(0.2), SEMICOLON, ... etc.

**Step 2 — Parser:** Builds a `Program` node with 7 child statement nodes:
- `LoadStatement(filename="iris.csv")`
- `NormalizeStatement(column_names=["sepal_length"])`
- `SplitStatement(train=0.8, test=0.2)`
- `ModelStatement(model_name="RandomForest", params={"trees": 100})`
- `TrainStatement(dataset="train_set")`
- `EvaluateStatement(result_var="accuracy", dataset="test_set")`
- `PrintStatement(expression=Variable("accuracy"), newline=True)`

**Step 3 — Evaluator:**
1. `eval_LoadStatement` → `ml_backend.load("iris.csv")` → `self.ml_backend.df` = 150-row iris DataFrame
2. `eval_NormalizeStatement` → `ml_backend.normalize(["sepal_length"])` → sets `ml_backend.normalization_columns = ["sepal_length"]`
3. `eval_SplitStatement` → `ml_backend.split(0.8, 0.2)`:
   - Splits data: X_train (120 rows), X_test (30 rows), y_train (120), y_test (30)
   - Fits StandardScaler on X_train["sepal_length"]
   - Transforms X_train["sepal_length"] in-place
   - Transforms X_test["sepal_length"] using same scaler (no fit)
   - Stores tuples in `ml_backend.train_set` and `ml_backend.test_set`
   - Evaluator binds: `variables["train_set"] = (X_train, y_train)`, `variables["test_set"] = (X_test, y_test)`
4. `eval_ModelStatement` → `ml_backend.set_model("RandomForest", {"trees": 100})`:
   - Translates `trees` → `n_estimators` via PARAM_MAP
   - Creates `RandomForestClassifier(n_estimators=100)`
5. `eval_TrainStatement` → looks up `variables["train_set"]` → `ml_backend.train((X_train, y_train))`:
   - `model.fit(X_train, y_train)` trains the forest
6. `eval_EvaluateStatement` → looks up `variables["test_set"]` → `ml_backend.evaluate((X_test, y_test))`:
   - `model.score(X_test, y_test)` returns accuracy (e.g., 0.9667)
   - Stores in `variables["accuracy"] = 0.9667`
   - Returns `0.9667`
7. `eval_PrintStatement` → `eval_Variable(Variable("accuracy"))` → `variables["accuracy"]` = `0.9667`
   - `print(0.9667, end='\n')` outputs the accuracy

**Output:** `0.9666666666666667`

---

## Test Suite Design

### test_evaluator.py — Core Language Tests

7 tests covering general-purpose language features:
- `==` and `!=` operators return correct Boolean results
- `if` executes the correct branch (true/false/no-else)
- `while` runs the correct number of iterations
- `while` with initially false condition never executes

### test_ml.py — ML Pipeline Tests

19 tests in 3 classes:

**TestMLHappyPath (11 tests):** Each test runs a complete FlowML ML pipeline and asserts that the accuracy score is in a reasonable range for that model on the iris dataset.
- RandomForest trees=100 → [0.85, 1.0]
- LogisticRegression → [0.7, 1.0]
- KNN neighbors=5 → [0.85, 1.0]
- SVM → [0.8, 1.0]
- NaiveBayes → [0.7, 1.0]
- Drop + LogisticRegression → [0.6, 1.0]
- Normalize + KNN → [0.85, 1.0]
- Drop + Normalize + RandomForest → [0.85, 1.0]
- Score stored and copied as variable → works correctly
- Ridge alpha=0.1 → is float (score range not guaranteed for regression)
- RandomForest trees=50 (param translation test) → [0.85, 1.0]

**TestMLErrorHandling (6 tests):** Each test asserts that a specific misuse raises an exception with an informative message.

**TestNormalizationDeferred (2 tests):** Confirms that deferred normalization works correctly (no data leakage) and that pipelines without normalization also work.

---

## Component 5: Environment (`environment.py`)

### Design

The `Environment` class implements lexical variable scoping for the evaluator. Each scope level is an independent `vars` dictionary linked to an optional `parent` scope. This forms a chain — looking up a variable walks up the chain until found or until the global (parentless) scope is reached.

### Why Use an Environment Instead of a Single Dictionary

Earlier versions of FlowML used a single flat dictionary (`self.variables`) in the Evaluator. This breaks for functions: a function must not permanently modify global variables that happen to share a name with a local variable. The `Environment` chain solves this:

- **Read:** Walk up the chain (`get`)
- **Write locally:** Write to the current scope only (`set`)
- **Write globally:** Walk to the root and write there (`set_global`) — used for function definitions and ML result bindings like `train_set`

### ReturnException

Function returns use Python's exception mechanism to unwind the call stack:

```python
class ReturnException(Exception):
    def __init__(self, value):
        self.value = value
```

When a `return` statement is evaluated, it raises `ReturnException(value)`. The function call handler (`eval_FunctionCall`) wraps the body execution in a `try/except ReturnException` block, extracts the value, restores the previous scope, and returns the value. This cleanly handles `return` from inside nested loops or conditionals without requiring special return-value passing through every loop/if evaluator.

### Scope Lifecycle for a Function Call

```
global_env
  vars: {x: 100, add: FunctionDefinition}

  eval_FunctionCall("add", [3, 7]):
    1. Evaluate args in global_env → [3, 7]
    2. Create local_env = Environment(parent=global_env)
    3. local_env.set("a", 3); local_env.set("b", 7)
    4. Execute body in local_env
       - "return a + b;" → raise ReturnException(10)
    5. Catch exception → result = 10
    6. Restore self.env = global_env
    7. Return 10
```

---

## Component 6: Symbol Table (`symbol_table.py`)

### HashTable

The `HashTable` is a fixed-size array of buckets with separate chaining (each bucket is a Python list). This is an implementation of the classic hash table taught in CS algorithms coursework.

**Hash function:** Polynomial rolling hash
```python
def _hash(self, key: str) -> int:
    h = 0
    for ch in key:
        h = (h * 31 + ord(ch)) % self.size
    return h
```

Using a prime multiplier (31) ensures that single-character keys and short strings produce well-distributed bucket indices rather than clustering.

**Collision resolution:** Separate chaining. When two keys hash to the same bucket index, both entries are appended to the same list. Lookup scans the list linearly. This is O(1) average, O(n) worst case.

**API:**
- `insert(name, dtype, value)` — adds new entry; raises `KeyError` if already present
- `update(name, value)` — updates value; raises `KeyError` if missing
- `lookup(name)` — returns copy of entry dict, or `None` if missing
- `delete(name)` — removes entry; raises `KeyError` if missing
- `contains(name)` — returns bool
- `all_entries()` — returns all entries by bucket traversal
- `display(label)` — prints a formatted table of all entries

### SymbolTable Wrapper

`SymbolTable` wraps `HashTable` with two additional responsibilities:

1. **Type inference:** On `declare(name, value)`, automatically infers the FlowML type string from the Python value type using `_infer_type()`:
   - `bool` → `"bool"` (checked before `int` since `bool` is a subclass of `int`)
   - `int` → `"int"`
   - `float` → `"float"`
   - `str` → `"str"`
   - `pd.DataFrame` → `"dataframe"`
   - `tuple` → `"dataset"` (used for `(X, y)` pairs from split)

2. **Strict type checking:** When `strict_types=True` (the default), calling `update(name, value)` with a value whose inferred type differs from the declared type raises `TypeError`. This catches programs like:
   ```
   x = 5;
   x = "hello";    // TypeError in strict mode: declared int, got str
   ```

The semantic analyzer uses `declare_type(name, dtype)` rather than `declare(name, value)` because it knows types at analysis time but not actual runtime values.

---

## Component 7: Semantic Analyzer (`semantic_analyzer.py`)

### Purpose

The semantic analyzer is an optional static pass that runs on the AST *before* execution. It traverses the same AST the evaluator would run, but instead of computing values, it tracks types and pipeline state, collecting errors into a list of `SemanticError` objects.

**Usage:**
```python
from flowml import analyze
errors = analyze(source, strict_types=True)
for e in errors:
    print(e)   # e.g., "SemanticError: Undefined variable 'y'"
```

### Design: Collect, Don't Stop

Unlike exceptions (which stop at the first error), the analyzer collects all errors it finds and returns them as a list. This lets the user see every problem in one pass, not just the first one.

### Checks Performed

**1. Undefined variable usage**
The analyzer maintains a `SymbolTable`. Every `AssignmentStatement` calls `declare()` or `update()`. Every `Variable` node checks `contains()`. If a variable is read before it has been declared, an error is recorded.

**2. Type mismatches (strict mode)**
When `strict_types=True`, reassigning a variable to a different type is flagged. The `SymbolTable.update()` method raises `TypeError` in strict mode; the analyzer catches this and records it as a `SemanticError`.

**3. Arithmetic type incompatibility**
For `BinaryExpression` nodes with arithmetic operators (`+`, `-`, `*`, `/`), the analyzer checks that both operands are numeric types (`"int"` or `"float"`). Mixed string/int arithmetic is flagged before execution.

**4. ML pipeline ordering**
The analyzer tracks five boolean flags to enforce the correct ML pipeline order:
```python
self._loaded    = False   # set True after load
self._split     = False   # set True after split
self._modeled   = False   # set True after model
self._trained   = False   # set True after train
self._evaluated = False   # set True after evaluate
```
Violations like `split` before `load`, `train` before `model`, or `evaluate` before `train` are all caught.

**5. Split ratio validation**
`SplitStatement.train + SplitStatement.test` is checked to equal `1.0`. Floating-point equality is handled with a tolerance.

**6. Unknown model names**
The model name in `ModelStatement` is checked against the same `MODEL_MAP` used by `MLBackend`. Unknown names are flagged.

### Limitation: Single-Pass, No Branching

The analyzer traverses statements sequentially and does not simulate both branches of `if/else`. This means it may not catch errors that only occur in one branch of a conditional. Full branch analysis would require a more sophisticated multi-pass or lattice-based type analysis.

---

## Component 8: Functions in the Evaluator

### FunctionDefinition

```python
def eval_FunctionDefinition(self, node: FunctionDefinition):
    self.env.set_global(node.name, node)
    return None
```

The function definition itself (the AST node) is stored as the value. This means function objects are first-class values in the environment — they can be looked up by name just like any variable.

### FunctionCall

The call evaluator implements the complete function call protocol:

1. Look up the function definition by name (raises exception if not a `FunctionDefinition`)
2. Evaluate all argument expressions in the *current* scope (before creating the child scope)
3. Validate argument count matches parameter count
4. Create a child `Environment` with the current scope as parent
5. Bind each parameter name to its argument value in the child scope
6. Execute the function body in the child scope; catch `ReturnException`
7. Restore the previous scope in a `finally` block (guarantees cleanup even if an exception propagates)
8. Return the caught return value, or `None` if no `return` was hit

### Recursion

Recursion works automatically. Each call creates a new child `Environment` with its own copy of local variables. The parent chain allows reading outer-scope variables (like the function definition itself), and the call stack is the Python call stack. Stack overflow for deeply recursive programs would raise Python's `RecursionError`.

---

## Updated Architecture: Entry Points

`__init__.py` now exposes two public entry points:

```python
from flowml.lexer import Lexer
from flowml.parser import Parser
from flowml.evaluator import Evaluator
from flowml.semantic_analyzer import SemanticAnalyzer

def interpret(source: str):
    """Full pipeline: lex → parse → evaluate."""
    tokens = Lexer(source).tokenize()
    ast    = Parser(tokens).parse()
    return Evaluator().evaluate(ast)

def analyze(source: str, strict_types: bool = True) -> list:
    """Lex → parse → semantic analysis only. Returns list of SemanticErrors."""
    tokens = Lexer(source).tokenize()
    ast    = Parser(tokens).parse()
    return SemanticAnalyzer(strict_types=strict_types).analyze(ast)
```

The semantic analysis pass is separate from execution and does not modify the AST. A program can pass semantic analysis and still fail at runtime (for errors the static pass cannot catch), and a program can be executed directly without running the semantic pass.
