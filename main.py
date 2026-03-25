import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python main.py <script.fml> [--compile] [--report]')
        sys.exit(1)

    source = open(sys.argv[1]).read()


    if '--compile' in sys.argv:
        output_path = sys.argv[1].replace('.fml', '.py')
        from flowml import compile_to_python
        python_code = compile_to_python(source)
        open(output_path, 'w').write(python_code)
        print(f'Compiled to {output_path}')

    else:
        from flowml import interpret
        interpret(source)
