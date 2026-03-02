from flowml import interpret

def main():

    # Test program

    with open("test.toy", "r") as f:
        source = f.read()
    
    print("Source code:")
    print(source)
    print("result:", interpret(source))

if __name__ == "__main__":
    main()  