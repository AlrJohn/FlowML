# FlowML Language Reference (Current)

Last updated: 2026-04-30

This document describes the current behavior of the FlowML DSL as implemented in this repository.

---

## 1. Overview

FlowML is a domain-specific language for writing machine learning pipelines and core control-flow logic in concise syntax.

FlowML supports three major workflows:

1. Interpret and execute a program.
2. Run static semantic analysis only.
3. Compile (transpile) FlowML source to standalone Python.

### 1.1 Execution Pipelines

Interpreter pipeline:

```text
.fml source -> Lexer -> Parser -> AST -> Evaluator -> runtime output
```

Semantic analysis pipeline:

```text
.fml source -> Lexer -> Parser -> AST -> SemanticAnalyzer -> SemanticError list
```

Compile pipeline:

```text
.fml source -> Lexer -> Parser -> AST -> SemanticAnalyzer -> CodeGenerator -> .py source
```

### 1.2 CLI Behavior (`main.py`)

- `python main.py program.fml`:
  runs semantic analysis first, then interprets only if there are no semantic errors.
- `python main.py program.fml --analyze`:
  runs semantic analysis only.
- `python main.py program.fml --compile`:
  runs semantic analysis and emits Python code when valid.

---

## 2. Lexical Rules

### 2.1 Comments

Single-line comments start with `//` and continue to end-of-line.

```flowml
// full line comment
x = 5; // trailing comment
```

### 2.2 Whitespace

Spaces, tabs, and newlines are token separators only.

### 2.3 Identifiers

Identifiers begin with a letter or underscore and continue with letters, digits, or underscores.

Examples:

```text
x
train_set
RandomForest
_temp1
```

### 2.4 Reserved Keywords

```text
print println if else while function return
true false
load drop normalize split model train test target
evaluate on into column columns data
```

### 2.5 Literals

- Integer: `42`, `0`, `7`
- Float: `0.8`, `3.14`, `1.0`
- String: `"iris.csv"`, `"species"`
- Boolean: `true`, `false`

String escapes currently supported by the lexer:

- `\n`
- `\t`
- `\\`
- `\"`

### 2.6 Operators and Punctuation

```text
+  -  *  /
==  !=  <  >  <=  >=
=
(  )  {  }
;  ,
```

---

## 3. Type Model

FlowML is dynamically typed at runtime, with additional static type categories used by the semantic analyzer.

### 3.1 Runtime Value Kinds

- Integer
- Float
- String
- Boolean
- DataFrame (from `load`)
- Dataset tuple `(X, y)` (from `split`)
- Model score float (from `evaluate`)

### 3.2 Semantic Analyzer Type Labels

The analyzer/symbol table track these type names:

- `int`
- `float`
- `str`
- `bool`
- `dataset`
- `dataframe`
- `model`
- `unknown`

---

## 4. Core Language Syntax

All statements end with a semicolon `;`.

### 4.1 Assignment

```flowml
x = 10;
score = accuracy;
result = (x + 5) * 2;
```

### 4.2 Output

- `print expr;` prints without newline.
- `println expr;` prints with newline.

```flowml
print "Accuracy: ";
println score;
```

### 4.3 Conditionals

```flowml
if (x > 5) {
    println "big";
} else {
    println "small";
}
```

Conditions use Python truthiness at runtime.

### 4.4 While Loops

```flowml
i = 0;
while (i < 3) {
    println i;
    i = i + 1;
}
```

### 4.5 Functions

Definition:

```flowml
function add(a, b) {
    return a + b;
}
```

Call as expression:

```flowml
x = add(3, 4);
println x;
```

Call as statement:

```flowml
add(1, 2);
```

Bare return is allowed:

```flowml
function f() {
    return;
}
```

---

## 5. Expressions and Precedence

### 5.1 Arithmetic and Unary

- `+`, `-`, `*`, `/`
- unary minus: `-x`

Important behavior:

- `/` currently performs integer floor-style division in evaluator/codegen (`//` behavior).

### 5.2 Comparisons

- `==`, `!=`, `<`, `>`, `<=`, `>=`

### 5.3 Precedence (low to high)

1. comparison operators
2. `+` and `-`
3. `*` and `/`
4. unary `-`
5. literals / variables / function calls / parenthesized expressions

### 5.4 Expression Grammar

```text
comparison = expression ((==|!=|<|>|<=|>=) expression)*
expression = term (('+'|'-') term)*
term       = unary (('*'|'/') unary)*
unary      = '-' unary | factor
factor     = NUMBER | FLOAT | STRING | BOOLEAN
           | IDENTIFIER
           | function_call
           | '(' expression ')'
```

---

## 6. Functions and Scope Semantics

Runtime evaluator scope behavior:

1. Function definitions are stored in global scope.
2. Each function call creates a child local scope.
3. Parameter bindings and assignments inside that call are local to that scope.
4. Reads can resolve through parent scope chain.
5. `return` uses an internal unwind mechanism and returns a value (or `None` for bare return).

Example:

```flowml
x = 100;

function demo(n) {
    x = n;       // local x
    return x;
}

println demo(42);  // 42
println x;         // 100
```

---

## 7. ML Statements

ML statements operate through the active backend state.

### 7.1 `load`

```flowml
load "iris.csv";
```

- Loads CSV into active DataFrame.

### 7.2 `drop columns`

```flowml
drop columns "sepal_width";
drop columns "c1", "c2";
```

- Removes one or more columns from active DataFrame.

### 7.3 `normalize columns`

```flowml
normalize columns "sepal_length", "petal_length";
```

- Deferred behavior: columns are recorded and scaling is applied during `split`.
- Prevents leakage by fitting scaler on training data then transforming test data.

### 7.4 `split`

```flowml
split data into train=0.8 test=0.2;
split data into train=0.8 test=0.2 target="species";
```

- Requires active DataFrame.
- Produces `train_set` and `test_set` variables, each `(X, y)`.
- `target="..."` is optional. If omitted, last column is used as target.

Ratio rule details:

- Semantic analyzer checks ratio with tolerance around 1.0.
- Runtime backend currently enforces exact `train + test == 1.0`.

### 7.5 `model`

```flowml
model RandomForest;
model RandomForest trees=100;
model KNN neighbors=5;
model Ridge alpha=0.1;
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

Parameter aliases:

- `trees` -> `n_estimators`
- `neighbors` -> `n_neighbors`
- `clusters` -> `n_clusters`
- `alpha` -> `alpha`

### 7.6 `train`

```flowml
train on train_set;
```

- Fits current model on referenced dataset tuple.

### 7.7 `evaluate`

```flowml
accuracy = evaluate on test_set;
println accuracy;
```

- Scores current model on referenced dataset tuple.
- Stores numeric result in assigned variable.

---

## 8. Static Semantic Analysis

The semantic analyzer walks the AST and returns a list of `SemanticError` objects.

### 8.1 Checks Currently Implemented

1. Undefined variable reads.
2. Assignment type mismatch checks (when `strict_types=True`).
3. Invalid operand type categories in arithmetic/comparison expressions.
4. Function call to undefined function.
5. Function argument count mismatch.
6. ML ordering checks:
   - `drop`, `normalize`, `split` before `load`
   - `train` before `model`
   - `train` before `split`
   - `evaluate` before `train`
7. Invalid split ratios.
8. Unknown model names.

### 8.2 Public API

```python
from flowml import analyze
errors = analyze(source_text, strict_types=True)
```

### 8.3 Compile-Path Note

`compile_to_python(...)` runs semantic analysis with `strict_types=False` before code generation.

---

## 9. Compilation and Code Generation

FlowML can generate standalone Python source code.

CLI:

```bash
python main.py script.fml --compile
```

Behavior:

1. Lex and parse source.
2. Run semantic checks.
3. If semantic errors exist, compilation aborts.
4. Otherwise emit Python code using pandas/scikit-learn imports.

Current generator coverage:

- assignments, print/println, if/else, while
- function definition/call/return
- load/drop/normalize/split/model/train/evaluate

Current generator caveat:

- `split ... target="..."` syntax exists in DSL/runtime, but codegen split emission currently always uses last column as target.

---

## 10. Formal Grammar (Current Parser Shape)

```text
program            ::= statement*

statement          ::= if_stmt
                     | while_stmt
                     | function_def
                     | return_stmt
                     | print_stmt
                     | println_stmt
                     | load_stmt
                     | drop_stmt
                     | normalize_stmt
                     | split_stmt
                     | model_stmt
                     | train_stmt
                     | evaluate_stmt
                     | assignment_stmt
                     | function_call ';'
                     | expression ';'

assignment_stmt    ::= IDENTIFIER '=' comparison ';'
evaluate_stmt      ::= IDENTIFIER '=' 'evaluate' 'on' IDENTIFIER ';'

if_stmt            ::= 'if' '(' comparison ')' block ('else' block)?
while_stmt         ::= 'while' '(' comparison ')' block
function_def       ::= 'function' IDENTIFIER '(' param_list? ')' block
return_stmt        ::= 'return' comparison? ';'

print_stmt         ::= 'print' comparison ';'
println_stmt       ::= 'println' comparison ';'

load_stmt          ::= 'load' STRING ';'
drop_stmt          ::= 'drop' 'columns' string_list ';'
normalize_stmt     ::= 'normalize' 'columns' string_list ';'

split_stmt         ::= 'split' 'data' 'into'
                       'train' '=' FLOAT
                       'test' '=' FLOAT
                       ('target' '=' STRING)?
                       ';'

model_stmt         ::= 'model' IDENTIFIER model_param_list? ';'
model_param_list   ::= (IDENTIFIER '=' model_param_value)+
model_param_value  ::= NUMBER | FLOAT | STRING | BOOLEAN

train_stmt         ::= 'train' 'on' IDENTIFIER ';'

block              ::= '{' statement* '}'
param_list         ::= IDENTIFIER (',' IDENTIFIER)*
arg_list           ::= comparison (',' comparison)*
string_list        ::= STRING (',' STRING)*

function_call      ::= IDENTIFIER '(' arg_list? ')'

comparison         ::= expression ((EQ|NEQ|LT|GT|LTE|GTE) expression)*
expression         ::= term (('+'|'-') term)*
term               ::= unary (('*'|'/') unary)*
unary              ::= '-' unary | factor
factor             ::= NUMBER | FLOAT | STRING | BOOLEAN
                     | IDENTIFIER
                     | function_call
                     | '(' expression ')'
```

---

## 11. Error Categories

Representative error categories:

- Lexer errors:
  unexpected character, unterminated string.
- Parser errors:
  unexpected token, missing delimiters/terminators.
- Semantic errors:
  undefined variables, type/category mismatches, ML ordering issues, bad split ratios, unknown models, function-call mismatches.
- Runtime evaluator/backend errors:
  division by zero, missing file, missing dataframe/model/dataset, unknown model, invalid ML operation order at runtime.

---

## 12. Current Limitations

The following are still not part of the language:

1. Boolean operators `and`, `or`, `not`.
2. `for` loops.
3. List/array literals and indexing syntax.
4. Object method call syntax in DSL.
5. `break` and `continue`.
6. Distinct float division operator (current `/` is integer-style floor behavior).

Static analysis limitations:

1. No full return-type inference for functions.
2. No deep path-sensitive branch typing.

---

## 13. Minimal End-to-End Example

```flowml
load "iris.csv";
normalize columns "sepal_length", "sepal_width", "petal_length", "petal_width";
split data into train=0.8 test=0.2 target="species";
model RandomForest trees=100;
train on train_set;
accuracy = evaluate on test_set;
println accuracy;
```

