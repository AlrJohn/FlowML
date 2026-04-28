import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python main.py <script.fml> [--compile] [--analyze]')
        sys.exit(1)

    filepath = sys.argv[1]
    try:
        source = open(filepath).read()
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)

    if '--compile' in sys.argv:
        from flowml import compile_to_python
        try:
            python_code  = compile_to_python(source)
            output_path  = filepath.replace('.fml', '.py').replace('.toy', '.py')
            open(output_path, 'w').write(python_code)
            print(f"Compiled successfully -> {output_path}")
        except Exception as e:
            print(f"Compilation failed: {e}")
            sys.exit(1)

    elif '--analyze' in sys.argv:
        from flowml import analyze
        errors = analyze(source)
        if errors:
            for error in errors:
                print(error)
        else:
            print("No semantic errors found.")

    else:
        # Default: run the semantic analyzer first, then interpret
        from flowml import analyze, interpret
        errors = analyze(source)
        if errors:
            for error in errors:
                print(error)
        else:
            interpret(source)
