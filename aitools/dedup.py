#!/usr/bin/env python3
"""
dedup.py - Advanced Python deduplication with AST parsing and Auto-Healing.

Features:
- Removes duplicate function/class definitions (keeps the LAST definition).
- Removes duplicate global imports (keeps the FIRST definition).
- Auto-detects and removes orphaned decorators that cause SyntaxErrors.
- Preserves file permissions and creates backups.
"""

import sys
import os
import ast
import shutil
import argparse
import tempfile
from collections import defaultdict

class CodeVisitor(ast.NodeVisitor):
    """
    Visits AST nodes to find function/class definitions and global imports.
    """
    def __init__(self):
        # Key: "full.path" -> List of (start_line, end_line)
        self.definition_locations = defaultdict(list)
        
        # Key: "import source string" -> List of (start_line, end_line)
        self.import_locations = defaultdict(list)
        
        self.current_path = []

    def _get_start_line(self, node):
        """
        Calculates the true start line, accounting for decorators.
        """
        start = node.lineno
        # Ensure we capture decorators if they exist
        if hasattr(node, 'decorator_list') and node.decorator_list:
            # The AST usually puts the first decorator at the earliest line
            start = min(d.lineno for d in node.decorator_list)
        return start

    def visit_Import(self, node):
        self._record_import(node)

    def visit_ImportFrom(self, node):
        self._record_import(node)

    def _record_import(self, node):
        """
        Records global imports. We only care about top-level imports 
        (scope is empty).
        """
        if not self.current_path:
            # We use the raw segment of code ideally, but AST doesn't give raw text easily.
            # We will use the line numbers to identify them later.
            # To distinguish 'duplicate' imports, we need a signature.
            if isinstance(node, ast.Import):
                sig = "import " + ", ".join(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ''
                names = ", ".join(a.name for a in node.names)
                sig = f"from {module} import {names}"
            else:
                return

            # Check for level (relative imports)
            if isinstance(node, ast.ImportFrom) and node.level:
                sig = "." * node.level + sig

            start = node.lineno
            # imports can be multi-line, but AST end_lineno covers that in Py3.8+
            end = getattr(node, 'end_lineno', node.lineno)
            
            self.import_locations[sig].append((start, end))

    def visit_FunctionDef(self, node):
        self._record_definition(node)

    def visit_AsyncFunctionDef(self, node):
        self._record_definition(node)

    def visit_ClassDef(self, node):
        # Add class to path, visit children, then pop
        self.current_path.append(node.name)
        
        # Record the class itself (handling decorators on the class)
        path_key = ".".join(self.current_path)
        start = self._get_start_line(node)
        end = node.end_lineno
        self.definition_locations[path_key].append((start, end))

        # Visit methods inside
        self.generic_visit(node)
        self.current_path.pop()

    def _record_definition(self, node):
        path_parts = self.current_path + [node.name]
        full_path = ".".join(path_parts)
        start = self._get_start_line(node)
        end = node.end_lineno
        
        self.definition_locations[full_path].append((start, end))
        # Do not recurse into functions (we don't dedup nested funcs usually)

def repair_source_code(source_code: str) -> str:
    """
    Iteratively attempts to parse the code. If a SyntaxError occurs
    on a line starting with '@', it removes that line (orphaned decorator)
    and retries.
    """
    max_retries = 20 # Prevent infinite loops
    current_source = source_code
    
    for _ in range(max_retries):
        try:
            ast.parse(current_source)
            return current_source # Parse successful
        except SyntaxError as e:
            if e.lineno is None:
                raise e # Can't fix generic errors without line numbers

            lines = current_source.splitlines(True)
            error_line_idx = e.lineno - 1
            
            if error_line_idx < len(lines):
                error_line = lines[error_line_idx]
                # Heuristic: If it's a SyntaxError on a decorator line, it's likely orphaned.
                if error_line.strip().startswith('@'):
                    print(f"  [Auto-Heal] Removing orphaned decorator at line {e.lineno}: {error_line.strip()}")
                    del lines[error_line_idx]
                    current_source = "".join(lines)
                    continue
            
            # If we get here, it's a SyntaxError we can't auto-fix safely
            print(f"Error: Syntax check failed at line {e.lineno}. Cannot auto-heal.", file=sys.stderr)
            raise e
            
    return current_source

def main():
    parser = argparse.ArgumentParser(description="Deduplicate Python functions/imports and heal orphaned decorators.")
    parser.add_argument("filename", help="The Python file to process")
    args = parser.parse_args()
    filename = args.filename

    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File '{filename}' not found")

    # Backup
    shutil.copy2(filename, f"{filename}.bak")
    print(f"Backup created: {filename}.bak")

    with open(filename, 'r', encoding='utf-8') as f:
        source_code = f.read()
        original_lines = source_code.splitlines(True)

    # 1. Initial Parse
    try:
        tree = ast.parse(source_code, filename=filename)
    except SyntaxError as e:
        raise SyntaxError(f"Original file has syntax errors. Aborting.\n{e}")

    visitor = CodeVisitor()
    visitor.visit(tree)

    lines_to_remove = set()

    # 2. Process Definitions (Keep LAST)
    dupe_defs = 0
    for path, locations in visitor.definition_locations.items():
        if len(locations) > 1:
            # Remove all but the last
            dupe_defs += (len(locations) - 1)
            for start, end in locations[:-1]:
                for i in range(start, end + 1):
                    lines_to_remove.add(i)

    # 3. Process Imports (Keep FIRST)
    dupe_imports = 0
    for sig, locations in visitor.import_locations.items():
        if len(locations) > 1:
            # Remove all but the first
            dupe_imports += (len(locations) - 1)
            for start, end in locations[1:]:
                for i in range(start, end + 1):
                    lines_to_remove.add(i)

    if not lines_to_remove:
        print("No duplicates found.")
        # We still run repair logic just in case the user has existing garbage
    else:
        print(f"Found {dupe_defs} duplicate definitions and {dupe_imports} redundant imports.")

    # 4. Reconstruct Source
    # Use enumerate 1-based to match AST
    new_lines = [line for i, line in enumerate(original_lines, 1) if i not in lines_to_remove]
    new_source = "".join(new_lines)

    # 5. Auto-Healing / Verification Loop
    print("Verifying syntax and checking for floating decorators...")
    try:
        final_source = repair_source_code(new_source)
    except SyntaxError as e:
        print("Failed to generate valid code. Original file restored.", file=sys.stderr)
        sys.exit(1)

    # 6. Write Output
    try:
        fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(filename), text=True)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(final_source)
        shutil.copymode(filename, tmp_path)
        shutil.move(tmp_path, filename)
        print(f"Success. File updated: {filename}")
    except Exception as e:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"Failed to write file: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal Error: {e}", file=sys.stderr)
        sys.exit(1)
