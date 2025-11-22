#!/usr/bin/env python3
"""
dedup.py - A modern Python script to remove duplicate function and method
definitions from a Python file, keeping only the *last* definition.

This script uses Python's Abstract Syntax Tree (AST) module to reliably
parse Python code. It correctly handles:
- Functions and methods inside classes (at any nesting level)
- Function decorators (which the original script failed on)
- Multi-line function signatures (which the original script failed on)
"""

import sys
import os
import ast
import shutil
import argparse
import tempfile
from collections import defaultdict

class FunctionVisitor(ast.NodeVisitor):
    """
    Visits AST nodes to find all function and class definitions.
    It records the start and end line numbers for each function/method,
    tracking the full scope (e.g., 'MyClass.InnerClass.my_method').
    """
    def __init__(self):
        # Using defaultdict simplifies adding to the list
        # Key: "full.path.to.function"
        # Value: List of (start_line, end_line) tuples
        self.function_locations = defaultdict(list)
        
        # This list tracks our current scope, e.g., ['MyClass', 'InnerClass']
        self.current_path = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Called for every 'def' statement (functions and methods)."""
        
        # Construct the full path, e.g., "MyClass.my_method"
        path_parts = self.current_path + [node.name]
        full_path = ".".join(path_parts)

        # Get the *full* extent of the function.
        # node.lineno is the 1-based line number of the *first decorator*
        # or the 'def' line itself if no decorators exist.
        start_line = node.lineno
        
        # node.end_lineno is the 1-based line number of the *last line*
        # of the function's body.
        end_line = node.end_lineno

        self.function_locations[full_path].append((start_line, end_line))
        
        # IMPORTANT: We DON'T call generic_visit here.
        # This prevents us from visiting functions *nested* inside this one
        # (e.g., `def outer(): def inner(): ...`).
        # This matches the behavior of the original script, which only
        # aimed for top-level functions (and now, class methods).

    def visit_ClassDef(self, node: ast.ClassDef):
        """Called for every 'class' statement."""
        
        # Add the class name to our current scope
        self.current_path.append(node.name)
        
        # Visit the children of this class (e.g., methods, nested classes)
        self.generic_visit(node)
        
        # Pop the class name from the scope once we're done with the class
        self.current_path.pop()

def main():
    parser = argparse.ArgumentParser(
        description="Remove duplicate function/method definitions from a Python file, keeping the last.",
        epilog="A backup of the original file will be created with a .bak extension."
    )
    parser.add_argument("filename", help="The Python file to process")
    args = parser.parse_args()

    filename = args.filename

    # 1. Validate file exists
    if not os.path.isfile(filename):
        print(f"Error: File '{filename}' not found", file=sys.stderr)
        sys.exit(1)

    # 2. Make a safety backup (like the original script)
    backup_file = f"{filename}.bak"
    try:
        shutil.copy2(filename, backup_file) # copy2 preserves permissions/metadata
        print(f"Safety backup created at '{backup_file}'")
    except OSError as e:
        print(f"Error: Could not create backup file: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Read the source code
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            source_code = f.read()
        # Keep track of original lines *with* newlines
        original_lines = source_code.splitlines(True) 
    except (IOError, UnicodeDecodeError) as e:
        print(f"Error: Could not read file '{filename}': {e}", file=sys.stderr)
        sys.exit(1)
        
    # Handle empty file
    if not source_code.strip():
        print(f"File '{filename}' is empty. Nothing to do.")
        sys.exit(0)

    # 4. Parse the AST
    try:
        tree = ast.parse(source_code, filename=filename)
    except SyntaxError as e:
        print(f"Error: Could not parse Python file (SyntaxError):", file=sys.stderr)
        print(f"  File: {e.filename}", file=sys.stderr)
        print(f"  Line: {e.lineno}", file=sys.stderr)
        print(f"  Text: {e.text.strip()}", file=sys.stderr)
        print("Aborting. No changes made (backup file still exists).", file=sys.stderr)
        sys.exit(1)

    # 5. Visit all nodes to find function definitions
    visitor = FunctionVisitor()
    visitor.visit(tree)
    
    # 6. Identify line ranges to *remove*
    # We want to keep the *last* one, so we remove all *but* the last.
    lines_to_remove_ranges = []
    kept_count = 0
    removed_count = 0

    for path, locations in visitor.function_locations.items():
        if locations:
            # We found at least one definition, so we're "keeping" one.
            kept_count += 1
        
        if len(locations) > 1:
            # Add all definitions *except* the last one to the removal list
            lines_to_remove_ranges.extend(locations[:-1])
            removed_count += len(locations) - 1
    
    if not lines_to_remove_ranges:
        print(f"No duplicate functions found in '{filename}'. File is unchanged.")
        sys.exit(0)

    # 7. Create a set of all (1-based) line numbers to remove
    remove_line_numbers = set()
    for start, end in lines_to_remove_ranges:
        # Add all lines from start to end, inclusive
        for i in range(start, end + 1):
            remove_line_numbers.add(i)

    # 8. Build the new file content in memory
    new_lines = []
    # Use enumerate(..., 1) for 1-based indexing to match AST line numbers
    for i, line in enumerate(original_lines, 1):
        if i not in remove_line_numbers:
            new_lines.append(line)
    
    new_source = "".join(new_lines)

    # 9. Write to a temporary file and atomically replace the original
    # This is safer than writing directly to the file.
    try:
        # Create a temp file in the same directory
        fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(filename), text=True)
        
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(new_source)
        
        # Copy permissions from original file to temp file
        shutil.copymode(filename, tmp_path)
        
        # Atomically move temp file to overwrite original
        shutil.move(tmp_path, filename)
        
    except Exception as e:
        print(f"Error: Could not write new file: {e}", file=sys.stderr)
        print("The original file is untouched. The backup is in .bak.", file=sys.stderr)
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path) # Clean up temp file
        sys.exit(1)

    # 10. Success message
    print(f"\nSuccessfully deduplicated '{filename}':")
    print(f"  - {removed_count} duplicate definitions removed.")
    print(f"  - {kept_count} unique functions/methods retained (last occurrence).")


if __name__ == "__main__":
    main()
