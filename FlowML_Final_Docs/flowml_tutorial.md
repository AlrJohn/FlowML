# FlowML Beginner Tutorial

This tutorial is a quick start for beginners.

You will learn:

1. How to write and run a basic `.fml` program.
2. How to build a simple ML pipeline in FlowML.

---

## 1. Before You Start

From the project root, make sure dependencies are installed:

```bash
pip install -r requirements.txt
```

FlowML files use the `.fml` extension.

---

## 2. Example 1: Basic Language Features

Create a file named `example1_basic.fml` with this code:

```flowml
// Example 1: variables, math, if/else, and while loop

x = 5;
y = 3;
sum = x + y;

println "Sum value:";
println sum;

if (sum > 7) {
    println "sum is greater than 7";
} else {
    println "sum is 7 or smaller";
}

i = 0;
while (i < 3) {
    println i;
    i = i + 1;
}
```

Run it:

```bash
python main.py example1_basic.fml
```

What this teaches:

- Variables are assigned with `=`.
- Every statement ends with `;`.
- Conditions use parentheses and blocks use `{ ... }`.
- `while` repeats until the condition becomes false.

---

## 3. Example 2: First ML Pipeline

Create a file named `example2_ml.fml` with this code:

```flowml
// Example 2: load data, split, train, evaluate

load "iris.csv";
split data into train=0.8 test=0.2 target="species";
model RandomForest trees=50;
train on train_set;
accuracy = evaluate on test_set;

println "Model accuracy:";
println accuracy;
```

Run it:

```bash
python main.py example2_ml.fml
```

What this teaches:

- `load` reads a CSV file.
- `split` creates `train_set` and `test_set`.
- `model` selects algorithm and optional parameters.
- `train` fits the model.
- `evaluate` returns a numeric score.

---

## 4. Helpful Commands While Learning

Semantic check only:

```bash
python main.py example2_ml.fml --analyze
```

Compile FlowML to Python:

```bash
python main.py example2_ml.fml --compile
```

This generates a `.py` file from your `.fml` program.

---

## 5. Beginner Tips

1. Start with tiny programs and add one statement at a time.
2. If execution fails, run `--analyze` first to catch semantic issues.
3. Keep dataset files in a path your program can access.
4. Use `println` often while learning to inspect intermediate values.
