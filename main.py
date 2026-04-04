import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python main.py <script.fml> [--compile] [--analyze]')
        sys.exit(1)

    source = open(sys.argv[1]).read()


    if '--compile' in sys.argv:
        pass # Compilation logic not implemented yet, but this is where it would go
        output_path = sys.argv[1].replace('.fml', '.py')
        from flowml import compile_to_python
        python_code = compile_to_python(source)
        open(output_path, 'w').write(python_code)
        print(f'Compiled to {output_path}')


    elif '--analyze' in sys.argv:
        from flowml import analyze
        errors = analyze(source)
        if errors:
            for error in errors:
                print(error)
        else:
            print("No semantic errors found.")

    else:
        #No flags provided, running interpreter by default. Use --compile or --analyze for other modes.
        from flowml import interpret
        from flowml import analyze
        
        errors = analyze(source)
        if errors:
            for error in errors:
                print(error)
        else:
            interpret(source)
