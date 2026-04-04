"""
symbol_table.py
---------------
Implements a HashTable using separate chaining (linked lists via Python lists)
and a SymbolTable wrapper that tracks variable names, data types, and values
for the FlowML interpreter.
"""


# ---------------------------------------------------------------------------
# HashTable
# ---------------------------------------------------------------------------

class HashTable:
    """
    A fixed-size hash table with separate chaining for collision resolution.

    Each bucket stores a list of entries:
        {"name": str, "dtype": str, "value": any}

    Hash function: polynomial rolling hash over character ordinals, mod size.
    """

    def __init__(self, size: int = 16):
        """
        Initialize the hash table with a fixed number of buckets. We use 16 by default because
        it provides a good balance between memory usage and collision probability for small programs.
        """

        self.size = size
        self.buckets: list[list[dict]] = [[] for _ in range(size)]
        self._count = 0

    # ---- internal ----

    def _hash(self, key: str) -> int:
        """Polynomial rolling hash so single-char keys don't all collide."""
        h = 0
        prime = 31 # A small prime number for the polynomial rolling hash
        for ch in key:
            h = (h * prime + ord(ch)) % self.size # Polynomial rolling hash, means each character contributes differently to the hash value
        return h

    def _find(self, name: str):
        """Return (bucket_index, entry_index, entry) or (idx, -1, None)."""
        idx = self._hash(name)
        for i, entry in enumerate(self.buckets[idx]):
            if entry["name"] == name:
                return idx, i, entry
        return idx, -1, None

    # ---- public API ----

    def insert(self, name: str, dtype: str, value) -> None:
        """Insert a new variable. Raises KeyError if the name/key already exists."""
        idx, pos, exist = self._find(name)
        if exist is not None:
            raise KeyError(f"Variable '{name}' already declared in symbol table.")
        self.buckets[idx].append({"name": name, "dtype": dtype, "value": value})
        self._count += 1

    def update(self, name: str, value) -> None:
        """Update the value of an existing variable. Raises KeyError if not found."""
        idx, pos, entry = self._find(name)
        if entry is None:
            raise KeyError(f"Variable '{name}' not found in symbol table.")
        self.buckets[idx][pos]["value"] = value

    def lookup(self, name: str) -> dict | None:
        """Return a copy of the entry dict, or None if not found."""
        _, _, entry = self._find(name)
        return dict(entry) if entry is not None else None

    def contains(self, name: str) -> bool:
        """Return True if the name is in the table."""
        _, _, entry = self._find(name)
        return entry is not None

    def delete(self, name: str) -> None:
        """Remove a variable. Raises KeyError if not found."""
        idx, pos, entry = self._find(name)
        if entry is None:
            raise KeyError(f"Variable '{name}' not found in symbol table.")
        self.buckets[idx].pop(pos)
        self._count -= 1

    def all_entries(self) -> list[dict]:
        """Return all entries in insertion order (by bucket traversal)."""
        entries = []
        for bucket in self.buckets:
            entries.extend(bucket)
        return entries

    def display(self, label: str = "Symbol Table") -> None:
        """Print every entry in a formatted table. This is mainly for debugging and demonstration purposes."""
        headers = ["Variable", "Type", "Value"]
        keys    = ["name",     "dtype", "value"]
        entries = self.all_entries()

        col_w = [
            max(len(h), max((len(str(e[k])) for e in entries), default=0))
            for h, k in zip(headers, keys)
        ]
        sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
        fmt = lambda vals: "| " + " | ".join(str(v).ljust(w) for v, w in zip(vals, col_w)) + " |"

        print(f"\n{'-' * (sum(col_w) + 10)}  {label}")
        print(sep)
        print(fmt(headers))
        print(sep)
        if not entries:
            print("| (empty)" + " " * (sum(col_w) + 7) + "|")
        else:
            for e in entries:
                print(fmt([e[k] for k in keys]))
        print(sep)

    def __len__(self) -> int:
        return self._count

    def __repr__(self) -> str:
        return f"HashTable(size={self.size}, entries={self._count})"


# ---------------------------------------------------------------------------
# SymbolTable
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    bool:  "bool",    # bool before int — bool is a subclass of int in Python
    int:   "int",
    float: "float",
    str:   "str",
    tuple: "dataset", # (X, y) pairs produced by split
}

# All type names that are valid in the symbol table (including ML types).
VALID_TYPES = {"bool", "int", "float", "str", "dataset", "dataframe", "model", "unknown"}


def _infer_type(value) -> str:
    """Infer FlowML type name from a Python value."""
    # Lazy import so pandas/sklearn are not required just to import symbol_table
    try:
        import pandas as pd
        if isinstance(value, pd.DataFrame):
            return "dataframe"
    except ImportError:
        pass
    for python_type, type_name in _TYPE_MAP.items():
        if isinstance(value, python_type):
            return type_name
    return "unknown"


class SymbolTable:
    """
    High-level symbol table backed by a HashTable.

    Responsibilities:
    - Declare variables with automatic type inference on first assignment
    - Validate that updates do not change the declared type (strict mode)
    - Print the full table state on demand (and automatically on updates)
    """

    def __init__(self, strict_types: bool = True):
        """
        Args:
            strict_types: If True, raise TypeError when a variable is
                          reassigned with a value of a different type.
        """
        self._table = HashTable()
        self.strict_types = strict_types

    # ---- variable lifecycle ----

    def declare(self, name: str, value) -> None:
        """
        Declare a new variable and infer its type from *value*.
        Raises KeyError if the variable has already been declared.
        """
        dtype = _infer_type(value)
        self._table.insert(name, dtype, value)

    def declare_type(self, name: str, dtype: str) -> None:
        """
        Declare a variable with an explicit type string and no value.
        Used by the semantic analyzer, which knows types at analysis time
        but not runtime values.
        Raises KeyError if already declared, ValueError for unknown type strings.
        """
        if dtype not in VALID_TYPES:
            raise ValueError(f"Unknown type '{dtype}'. Valid types: {VALID_TYPES}")
        self._table.insert(name, dtype, None)

    def update(self, name: str, value) -> None:
        """
        Update an existing variable's value.

        If strict_types is True, raises TypeError when the new value's
        inferred type differs from the declared type.

        Prints the symbol table BEFORE and AFTER the update.
        """
        entry = self._table.lookup(name)
        if entry is None:
            raise KeyError(f"Undefined variable '{name}'.")

        new_dtype = _infer_type(value)
        if self.strict_types and new_dtype != entry["dtype"]:
            raise TypeError(
                f"Type mismatch for '{name}': "
                f"declared as '{entry['dtype']}', got '{new_dtype}'."
            )

        self.print_state(f"BEFORE update: {name} = {value}")
        self._table.update(name, value)
        self.print_state(f"AFTER  update: {name} = {value}")

    def get(self, name: str):
        """Return the current value of a variable. Raises KeyError if undefined."""
        entry = self._table.lookup(name)
        if entry is None:
            raise KeyError(f"Undefined variable '{name}'.")
        return entry["value"]

    def get_dtype(self, name: str) -> str:
        """Return the declared type of a variable."""
        entry = self._table.lookup(name)
        if entry is None:
            raise KeyError(f"Undefined variable '{name}'.")
        return entry["dtype"]

    def contains(self, name: str) -> bool:
        return self._table.contains(name)

    # ---- display ----

    def print_state(self, label: str = "Symbol Table State") -> None:
        """Print the current symbol table."""
        self._table.display(label)

    def __len__(self) -> int:
        return len(self._table)

    def __repr__(self) -> str:
        return f"SymbolTable(vars={len(self._table)}, strict={self.strict_types})"


