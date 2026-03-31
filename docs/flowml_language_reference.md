# FlowML DSL: Complete Language Reference

## Overview

FlowML is a Domain-Specific Language (DSL) designed for machine learning pipelines. It provides a high-level, declarative syntax that abstracts away the complexity of traditional Python ML libraries such as scikit-learn, pandas, and numpy. The goal is to allow users to write complete ML workflows — from loading data to evaluating a trained model — in a concise, readable, domain-specific manner.

FlowML programs are interpreted by a tree-walking interpreter implemented in Python. The interpreter pipeline is:

```
Source Code (.fml) → Lexer → Token Stream → Parser → AST → Evaluator → Output / ML Results
```

FlowML files conventionally use the `.fml` extension, though `.toy` is also used for experimental scripts.

---

## 1. Lexical Rules

### 1.1 Comments

FlowML supports single-line comments using `//`. Everything from `//` to the end of the line is ignored.

```
// This is a comment
x = 5; // inline comment
```

### 1.2 Whitespace

Spaces, tabs, and newlines are ignored between tokens. They serve only as separators.

### 1.3 Identifiers

Identifiers begin with a letter (`a-z`, `A-Z`) or underscore (`_`), followed by any combination of letters, digits, or underscores. Identifiers are used for variable names and model names.

Examples: `x`, `accuracy`, `train_set`, `sepal_length`, `RandomForest`

### 1.4 Keywords

The following words are reserved keywords and cannot be used as identifiers:

```
print  println  if  else  while  function  return
true  false
load  drop  normalize  split  model  train  test
evaluate  on  into  column  columns  data
```

### 1.5 Literals

**Integer:** A sequence of one or more digits.
```
5    42    0    100
```

**Float:** Digits, a decimal point, and more digits. At least one digit must appear after the decimal.
```
3.14    0.8    0.2    1.0
```

**String:** Characters enclosed in double quotes. Escape sequences supported:
- `\n` → newline
- `\t` → tab
- `\\` → backslash
- `\"` → double quote

```
"iris.csv"    "sepal_length"    "Hello, World!\n"
```

**Boolean:** The literals `true` and `false`.

### 1.6 Operators and Punctuation

```
+  -  *  /          Arithmetic operators
==  !=  <  >  <=  >= Comparison operators
=                    Assignment
(  )                 Parentheses
{  }                 Block delimiters
;                    Statement terminator
,                    List separator
```

---

## 2. Data Types

FlowML is dynamically typed. Variables do not require type declarations. The following value types exist at runtime:

| Type | Description | Examples |
|------|-------------|---------|
| Integer | Whole numbers | `5`, `0`, `-10` |
| Float | Decimal numbers | `3.14`, `0.8` |
| String | Text values | `"hello"`, `"iris.csv"` |
| Boolean | Truth values | `true`, `false` |
| DataFrame | Pandas DataFrame from CSV | (loaded via `load`) |
| DatasetTuple | Pair `(X, y)` from split | (bound as `train_set`, `test_set`) |
| Float (score) | Model accuracy | (returned by `evaluate`) |

---

## 3. Variables and Assignment

Variables are dynamically typed and scoped globally (single scope — no block scoping).

**Syntax:**
```
variable_name = expression;
```

**Examples:**
```
x = 10;
pi = 3.14159;
label = "accuracy";
flag = true;
result = 5 + 3 * 2;
copy = accuracy;      // copy another variable's value
```

Variables are stored in the evaluator's `self.variables` dictionary. They persist for the entire program's lifetime. Reading an undefined variable raises a runtime error.

---

## 4. Expressions

### 4.1 Arithmetic Expressions

FlowML supports standard arithmetic with proper operator precedence.

| Operator | Description | Precedence |
|----------|-------------|------------|
| `+` | Addition | Low |
| `-` | Subtraction | Low |
| `*` | Multiplication | Medium |
| `/` | Integer Division | Medium |
| `-x` | Unary negation | High |

Note: Division (`/`) currently performs integer (floor) division.

**Grammar:**
```
expression = term (('+' | '-') term)*
term       = unary (('*' | '/') unary)*
unary      = '-' unary | factor
factor     = NUMBER | FLOAT | STRING | BOOLEAN | IDENTIFIER | '(' expression ')'
```

**Examples:**
```
result = 5 + 3 * 2;       // 11 (multiplication first)
calc = (5 + 3) * 2;       // 16 (parens override precedence)
neg = -10;                 // -10
div = 7 / 2;              // 3 (integer division)
```

### 4.2 Comparison Expressions

Comparison expressions return a Boolean value (`true` or `false`).

| Operator | Description |
|----------|-------------|
| `==` | Equal to |
| `!=` | Not equal to |
| `<` | Less than |
| `>` | Greater than |
| `<=` | Less than or equal |
| `>=` | Greater than or equal |

Comparisons have lower precedence than arithmetic. An arithmetic expression is evaluated first, and then the comparison is applied.

**Grammar:**
```
comparison = expression ((==|!=|<|>|<=|>=) expression)*
```

**Examples:**
```
5 == 5;       // true
5 != 3;       // true
x > 5;        // depends on x
i < 10;       // depends on i
```

---

## 5. Statements

All statements must end with a semicolon `;`.

### 5.1 Assignment

```
variable = expression;
```

Evaluates the right-hand expression and stores the result in the named variable.

### 5.2 Print Statements

**`print`** — Outputs a value without a trailing newline.
**`println`** — Outputs a value followed by a newline.

```
print "Hello ";
println "World";      // Hello World (with newline at end)
println x;
println accuracy;
```

Both statements accept any expression.

### 5.3 If / Else Statement

Conditionally executes a block of statements.

**Syntax:**
```
if (condition) {
    statements
}

if (condition) {
    statements
} else {
    statements
}
```

- The condition is any comparison or expression (truthiness is used)
- The `else` branch is optional
- Both branches use block syntax `{ ... }`

**Examples:**
```
x = 3;
if (x > 5) {
    println "big";
} else {
    println "small";
}
// Prints: small

if (accuracy) {
    println accuracy;
}
```

### 5.4 While Loop

Repeatedly executes a block while the condition remains true.

**Syntax:**
```
while (condition) {
    statements
}
```

- Condition is re-evaluated before every iteration
- Terminates when condition is false (or was never true)
- No break or continue keywords (not yet implemented)

**Examples:**
```
i = 0;
while (i < 5) {
    println i;
    i = i + 1;
}
// Prints: 0, 1, 2, 3, 4

x = 10;
while (x > 0) {
    println x;
    x = x - 1;
}
// Counts down from 10 to 1
```

---

## 6. ML Data Operations

These statements operate on the active DataFrame managed by the MLBackend.

### 6.1 load

Loads a CSV file into memory as the active DataFrame.

**Syntax:**
```
load "path/to/file.csv";
```

- Path is relative to the working directory where the interpreter is run
- The CSV's last column is treated as the target (label) variable
- All other columns are treated as features
- Internally uses `pandas.read_csv()`

**Example:**
```
load "iris.csv";
```

### 6.2 drop columns

Removes one or more columns from the active DataFrame.

**Syntax:**
```
drop columns "col1";
drop columns "col1", "col2", "col3";
```

- Columns are specified as comma-separated string literals
- Raises an error if the column does not exist
- Useful for feature selection/elimination

**Example:**
```
load "iris.csv";
drop columns "sepal_width";
```

### 6.3 normalize columns

Marks columns for StandardScaler normalization. Normalization is **deferred** — it does not happen immediately. Instead, the columns are recorded and applied during `split`, separately to the training and test sets. This is a critical design choice that **prevents data leakage**.

**Syntax:**
```
normalize columns "col1";
normalize columns "col1", "col2", "col3";
```

- Columns are specified as comma-separated string literals
- Normalization uses `sklearn.preprocessing.StandardScaler`
- Scaler is fit only on training data (`fit_transform`), then applied to test data (`transform`)
- Calling `normalize` a second time overwrites the pending column list

**Example:**
```
load "iris.csv";
normalize columns "sepal_length", "sepal_width", "petal_length", "petal_width";
split data into train=0.8 test=0.2;
// Normalization is applied here, after the split
```

### 6.4 split

Splits the active DataFrame into training and test sets.

**Syntax:**
```
split data into train=<float> test=<float>;
```

- `train` and `test` values must be floats that sum to exactly `1.0`
- Raises an error if ratios do not sum to 1.0
- Last column is automatically used as the target
- All other columns are features
- Creates two variables automatically: `train_set` and `test_set`
- Each is a tuple `(X, y)` where `X` is a DataFrame of features and `y` is a Series of labels
- Internally uses `sklearn.model_selection.train_test_split()`
- Any deferred normalization is applied here

**Example:**
```
split data into train=0.8 test=0.2;
// Creates: train_set = (X_train, y_train), test_set = (X_test, y_test)
```

---

## 7. ML Model Operations

### 7.1 model

Instantiates an ML model with optional parameters.

**Syntax:**
```
model ModelName;
model ModelName param1=value1 param2=value2;
```

**Supported Models:**

| FlowML Name | scikit-learn Class | Type |
|-------------|-------------------|------|
| `LogisticRegression` | `LogisticRegression` | Classification |
| `RandomForest` | `RandomForestClassifier` | Classification |
| `SVM` | `SVC` | Classification |
| `KNN` | `KNeighborsClassifier` | Classification |
| `NaiveBayes` | `GaussianNB` | Classification |
| `LinearRegression` | `LinearRegression` | Regression |
| `Ridge` | `Ridge` | Regression |
| `Lasso` | `Lasso` | Regression |
| `KMeans` | `KMeans` | Clustering |

**Parameter Mapping (FlowML → scikit-learn):**

| FlowML Param | scikit-learn Param | Used By |
|-------------|-------------------|---------|
| `trees` | `n_estimators` | RandomForest |
| `neighbors` | `n_neighbors` | KNN |
| `clusters` | `n_clusters` | KMeans |
| `alpha` | `alpha` | Ridge, Lasso |

Other parameter names are passed through directly to scikit-learn.

**Examples:**
```
model RandomForest trees=100;
model KNN neighbors=5;
model Ridge alpha=0.1;
model LogisticRegression;       // No params — uses sklearn defaults
model SVM;
model NaiveBayes;
model KMeans clusters=3;
```

### 7.2 train

Trains the current model on a dataset.

**Syntax:**
```
train on dataset_variable;
```

- `dataset_variable` must be a variable holding a `(X, y)` tuple (typically `train_set`)
- Calls `model.fit(X, y)` internally
- If the model is a regressor and the target labels are categorical strings, they are automatically encoded as integers before fitting
- Raises an error if no model has been defined, or if the variable is undefined

**Example:**
```
train on train_set;
```

### 7.3 evaluate

Evaluates the trained model on a dataset and stores the score in a variable.

**Syntax:**
```
result_variable = evaluate on dataset_variable;
```

- `dataset_variable` must be a variable holding a `(X, y)` tuple (typically `test_set`)
- Returns the model's accuracy score (0.0 to 1.0) using `model.score(X, y)`
- Stores the result in `result_variable` for later use
- Raises an error if no model is defined, or if the variable is undefined

**Example:**
```
accuracy = evaluate on test_set;
println accuracy;     // e.g., 0.9666666666666667
```

---

## 8. Complete Example Programs

### Example 1: Simple Countdown (test_conditionals.fml)
```
x = 10;
while (x > 0) {
    println x;
    x = x - 1;
}
// Prints 10, 9, 8, ..., 1
```

### Example 2: ML Pipeline in a Loop (test.toy)
```
i = 5;
while (i > 0) {
    load "iris.csv";
    normalize columns "sepal_length", "sepal_width", "petal_length", "petal_width";
    split data into train=0.8 test=0.2;
    model RandomForest trees=100;
    train on train_set;
    accuracy = evaluate on test_set;
    if (accuracy) {
        println accuracy;
    }
    i = i - 1;
}
// Runs a full ML pipeline 5 times, printing accuracy each time
```

### Example 3: Basic Iris Classification
```
load "iris.csv";
split data into train=0.8 test=0.2;
model RandomForest trees=100;
train on train_set;
accuracy = evaluate on test_set;
println accuracy;
```

### Example 4: Drop + Normalize + KNN
```
load "iris.csv";
drop columns "sepal_width";
normalize columns "sepal_length", "petal_length", "petal_width";
split data into train=0.8 test=0.2;
model KNN neighbors=5;
train on train_set;
score = evaluate on test_set;
println score;
```

### Example 5: Ridge Regression
```
load "iris.csv";
split data into train=0.8 test=0.2;
model Ridge alpha=0.1;
train on train_set;
score = evaluate on test_set;
println score;
```

### Example 6: Using evaluate result in logic
```
load "iris.csv";
split data into train=0.8 test=0.2;
model LogisticRegression;
train on train_set;
accuracy = evaluate on test_set;
copy = accuracy;
println copy;
```

---

## 9. Formal Grammar (BNF)

```
program       ::= statement*

statement     ::= assignment_stmt
                | if_stmt
                | while_stmt
                | print_stmt
                | println_stmt
                | load_stmt
                | drop_stmt
                | normalize_stmt
                | split_stmt
                | model_stmt
                | train_stmt
                | evaluate_stmt
                | expression ';'

assignment_stmt    ::= IDENTIFIER '=' comparison ';'
if_stmt            ::= 'if' '(' comparison ')' block ( 'else' block )?
while_stmt         ::= 'while' '(' comparison ')' block
print_stmt         ::= 'print' comparison ';'
println_stmt       ::= 'println' comparison ';'

load_stmt          ::= 'load' STRING ';'
drop_stmt          ::= 'drop' 'columns' string_list ';'
normalize_stmt     ::= 'normalize' 'columns' string_list ';'
split_stmt         ::= 'split' 'data' 'into' 'train' '=' FLOAT 'test' '=' FLOAT ';'
model_stmt         ::= 'model' IDENTIFIER param_list? ';'
train_stmt         ::= 'train' 'on' IDENTIFIER ';'
evaluate_stmt      ::= IDENTIFIER '=' 'evaluate' 'on' IDENTIFIER ';'

block              ::= '{' statement* '}'
param_list         ::= (IDENTIFIER '=' factor)+
string_list        ::= STRING (',' STRING)*

comparison         ::= expression ((EQ|NEQ|LT|GT|LTE|GTE) expression)*
expression         ::= term (('+' | '-') term)*
term               ::= unary (('*' | '/') unary)*
unary              ::= '-' unary | factor
factor             ::= NUMBER | FLOAT | STRING | BOOLEAN | IDENTIFIER | '(' expression ')'
```

---

## 10. Error Handling

FlowML provides informative error messages for common errors:

| Error | Example | Message |
|-------|---------|---------|
| Lexer: unknown character | `@x = 5;` | `Lexer Error at 1:1: Unexpected character: '@'` |
| Lexer: unterminated string | `"hello` | `Lexer Error at 1:1: Unterminated string literal` |
| Parser: unexpected token | `x = ;` | `Parser Error at 1:5: Expected ... but got SEMICOLON` |
| Runtime: undefined variable | `println y;` | `Evaluator: Undefined variable 'y'` |
| Runtime: division by zero | `x = 5 / 0;` | `Evaluator: Division by zero` |
| Runtime: unknown model | `model FakeModel;` | `MLBackend: Unknown model type 'FakeModel'` |
| Runtime: split ratio error | `split data into train=0.8 test=0.3;` | `MLBackend: Train and test ratios must sum to 1` |
| Runtime: train before model | `train on train_set;` (no model) | `MLBackend: No model defined. Call 'model' before 'train'.` |
| Runtime: file not found | `load "missing.csv";` | File not found error from pandas |
| Runtime: drop missing column | `drop columns "fake";` | `MLBackend: Column 'fake' does not exist` |
| Runtime: no DataFrame | `drop columns "x";` (no load) | `Evaluator: No active DataFrame to drop column from` |

---

## 11. Current Limitations

The following features are **not yet implemented** in the current version:

- **Function definitions:** `function` and `return` tokens are defined in the lexer but the parser and evaluator do not handle function declarations yet.
- **Boolean operators:** `and`, `or`, `not` are not supported. Conditions can only be simple comparisons.
- **For loops:** No `for` construct exists. Iteration requires `while` with a counter.
- **Arrays/Lists:** No array or list data type. No indexing syntax.
- **String operations:** No string concatenation via `+`, no string length or slicing.
- **Method calls:** No object method call syntax (e.g., `df.describe()`).
- **Compilation to Python:** The `--compile` flag and `compile_to_python()` function are referenced but not yet implemented.
- **Float division:** The `/` operator currently performs integer (floor) division. True float division is not yet supported.
- **Multiple datasets:** Only one active DataFrame at a time. Multiple datasets require re-loading.
