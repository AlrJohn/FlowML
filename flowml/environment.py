"""
FlowML Environment

Manages variable scope for the FlowML evaluator.

Each Environment has its own variable store (vars) and an optional
reference to a parent Environment. When a variable is looked up,
the search starts in the current scope and walks up the parent chain
until found or until there is no parent (global scope).

Usage:
    # Global scope
    global_env = Environment()

    # Function scope — child of global
    local_env = Environment(parent=global_env)

    local_env.set('x', 10)     # sets in local scope
    local_env.get('x')         # finds in local scope -> 10
    local_env.get('y')         # not in local, walks up to global
"""


class ReturnException(Exception):
    """
    Raised by return statements to unwind the call stack.
    Caught by eval_FunctionCall to extract the return value.
    Using an exception is the cleanest way to exit from any
    depth of nested loops or conditionals inside a function body.
    """
    def __init__(self, value):
        self.value = value


class Environment:
    """Holds variable bindings for a single scope level."""

    def __init__(self, parent=None):
        self.vars   = {}
        self.parent = parent

    def get(self, name: str):
        """
        Look up a variable name in this scope or any parent scope.
        Raises NameError if not found anywhere in the chain.
        """
        if name in self.vars:
            return self.vars[name]
        if self.parent:
            return self.parent.get(name)
        raise Exception(f"Undefined variable '{name}'")

    def set(self, name: str, value):
        """
        Set a variable in the current (innermost) scope.
        Does not walk up to parent — assignment always
        creates or updates in the current scope.
        """
        self.vars[name] = value

    def set_global(self, name: str, value):
        """
        Set a variable in the outermost (global) scope.
        Used to store function definitions at the top level
        regardless of where the definition appears.
        """
        if self.parent is None:
            self.vars[name] = value
        else:
            self.parent.set_global(name, value)