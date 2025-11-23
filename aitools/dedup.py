#!/usr/bin/env python3
"""
dedup.py - Advanced Python deduplication with Class Merging, AST parsing, and Auto-Healing.

Features:
- Merges duplicate Class definitions:
    - Designates the LAST class definition as the "Master".
    - Moves unique methods from earlier definitions into the Master.
    - Deletes the earlier class definitions.
- Removes duplicate global function definitions (keeps the LAST definition).
- Removes duplicate global imports (keeps the FIRST definition).
- Auto-detects and removes orphaned decorators that cause SyntaxErrors.
- Cleans up excessive vertical whitespace.
- Preserves file permissions and creates backups.
"""

import sys
import os
import ast
import shutil
import argparse
import tempfile
import re
from collections import defaultdict

# ---------------------------------------------------------------------
# AST Analyzers
# ---------------------------------------------------------------------

class ClassInfo:
    def __init__(self, node):
        self.node = node
        self.name = node.name
        self.methods = {} # name -> node
        self.start = 0
        self.end = 0
        self.body_start_line = 0 

    def analyze_methods(self):
        """Builds a dictionary of method names to nodes."""
        for child in self.node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.methods[child.name] = child

class ClassMergeAnalyzer(ast.NodeVisitor):
    """
    Identifies classes that need to be merged.
    Strategy:
    1. Find all definitions of class X.
    2. The LAST definition is the 'Master'.
    3. Any methods in previous definitions of X that are NOT in Master
       must be extracted and injected into Master.
    4. Previous definitions are marked for total deletion.
    """
    def __init__(self, source_lines):
        self.classes = defaultdict(list) # name -> list of ClassInfo
        self.source_lines = source_lines
        self.lines_to_remove = set()
        self.injections = defaultdict(list) # line_idx -> list of strings (code to insert)

    def _get_true_start(self, node):
        """Calculates start line including decorators."""
        start = node.lineno
        if hasattr(node, 'decorator_list') and node.decorator_list:
            start = min(d.lineno for d in node.decorator_list)
        return start

    def visit_ClassDef(self, node):
        # We only handle top-level classes for safety in merging logic
        if getattr(node, 'col_offset', 0) == 0:
            info = ClassInfo(node)
            info.start = self._get_true_start(node)
            info.end = node.end_lineno
            info.analyze_methods()
            self.classes[node.name].append(info)
        self.generic_visit(node)

    def process_merges(self):
        """
        Calculates deletions and injections. 
        Returns a set of class names that were processed (merged), 
        so the standard dedup visitor ignores them.
        """
        processed_classes = set()

        for name, info_list in self.classes.items():
            if len(info_list) < 2:
                continue

            processed_classes.add(name)
            
            # The last definition is the Master
            master = info_list[-1]
            
            # Identify the indentation of the master class body to match it
            # We guess 4 spaces if we can't find it, or look at the first method
            indent_str = "    "
            if master.methods:
                first_method = list(master.methods.values())[0]
                # AST cols are 0-indexed.
                indent_col = first_method.col_offset
                indent_str = " " * indent_col

            for prev_class in info_list[:-1]:
                # 1. Mark previous class for full deletion
                for i in range(prev_class.start, prev_class.end + 1):
                    self.lines_to_remove.add(i)

                # 2. Scan for unique methods to teleport
                for method_name, method_node in prev_class.methods.items():
                    if method_name not in master.methods:
                        # This method exists in the old class but not the new one.
                        # We must keep it.
                        
                        # Extract source text
                        m_start = self._get_true_start(method_node)
                        m_end = method_node.end_lineno
                        
                        # Python lists are 0-indexed, line numbers are 1-indexed
                        raw_lines = self.source_lines[m_start-1 : m_end]
                        
                        # De-indent logic (the old class might have different indent depth?)
                        # Assuming top-level classes, indent should be standard, but let's be safe.
                        # We'll just take the raw lines. If both are top level, indentation matches.
                        
                        code_chunk = "".join(raw_lines)
                        # Ensure whitespace separation
                        code_chunk = f"\n{code_chunk}\n"
                        
                        # Inject at the end of the Master class
                        # We inject specifically at the last line of the Master class
                        self.injections[master.end].append(code_chunk)

        return processed_classes


class StandardDedupVisitor(ast.NodeVisitor):
    """
    Standard deduplication for global functions and imports.
    Ignores classes that are being handled by the MergeAnalyzer.
    """
    def __init__(self, ignore_classes):
        self.definition_locations = defaultdict(list)
        self.import_locations = defaultdict(list)
        self.ignore_classes = ignore_classes
        self.current_scope = []

    def _get_start_line(self, node):
        start = node.lineno
        if hasattr(node, 'decorator_list') and node.decorator_list:
            start = min(d.lineno for d in node.decorator_list)
        return start

    def visit_Import(self, node):
        self._record_import(node)

    def visit_ImportFrom(self, node):
        self._record_import(node)

    def _record_import(self, node):
        # Only global imports
        if not self.current_scope:
            if isinstance(node, ast.Import):
                sig = "import " + ", ".join(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ''
                names = ", ".join(a.name for a in node.names)
                sig = f"from {module} import {names}"
            else:
                return

            if isinstance(node, ast.ImportFrom) and node.level:
                sig = "." * node.level + sig

            start = node.lineno
            end = getattr(node, 'end_lineno', node.lineno)
            self.import_locations[sig].append((start, end))

    def visit_FunctionDef(self, node):
        # Only global functions
        if not self.current_scope:
            self._record_definition(node)

    def visit_AsyncFunctionDef(self, node):
        if not self.current_scope:
            self._record_definition(node)

    def visit_ClassDef(self, node):
        # If this class is being merged, skip logic here
        if node.name in self.ignore_classes:
            return

        self.current_scope.append(node.name)
        self._record_definition(node)
        self.generic_visit(node)
        self.current_scope.pop()

    def _record_definition(self, node):
        path_parts = self.current_scope + [node.name]
        full_path = ".".join(path_parts)
        start = self._get_start_line(node)
        end = node.end_lineno
        self.definition_locations[full_path].append((start, end))


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def clean_whitespace(source_code: str) -> str:
    """
    Collapses 3 or more consecutive newlines into 2.
    Removes trailing whitespace on empty lines.
    """
    # 1. Consolidate vertical whitespace (max 2 blank lines)
    # Regex: \n{3,} matches 3 or more newlines. Replace with \n\n.
    source_code = re.sub(r'\n{3,}', '\n\n\n', source_code)
    
    return source_code

def repair_source_code(source_code: str) -> str:
    """
    Iteratively attempts to parse the code to fix orphaned decorators.
    """
    max_retries = 20 
    current_source = source_code
    
    for _ in range(max_retries):
        try:
            ast.parse(current_source)
            return current_source
        except SyntaxError as e:
            if e.lineno is None:
                raise e 

            lines = current_source.splitlines(True)
            error_line_idx = e.lineno - 1
            
            if error_line_idx < len(lines):
                error_line = lines[error_line_idx]
                # Heuristic: SyntaxError on decorator line -> orphaned.
                if error_line.strip().startswith('@'):
                    print(f"  [Auto-Heal] Removing orphaned decorator at line {e.lineno}: {error_line.strip()}")
                    del lines[error_line_idx]
                    current_source = "".join(lines)
                    continue
            
            print(f"Error: Syntax check failed at line {e.lineno}. Cannot auto-heal.", file=sys.stderr)
            raise e
            
    return current_source

def main():
    parser = argparse.ArgumentParser(description="Deduplicate Python code, merge classes, and clean whitespace.")
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

    lines_to_remove = set()
    injections = defaultdict(list)

    # 2. Run Class Merger (Pass 1)
    print("Analyzing class structures for merging...")
    merger = ClassMergeAnalyzer(original_lines)
    merger.visit(tree)
    
    merged_class_names = merger.process_merges()
    
    lines_to_remove.update(merger.lines_to_remove)
    for k, v in merger.injections.items():
        injections[k].extend(v)

    if merged_class_names:
        print(f"Merging {len(merged_class_names)} classes: {', '.join(merged_class_names)}")

    # 3. Run Standard Dedup (Pass 2)
    dedup_visitor = StandardDedupVisitor(ignore_classes=merged_class_names)
    dedup_visitor.visit(tree)

    # Process standard defs
    dupe_defs = 0
    for path, locations in dedup_visitor.definition_locations.items():
        if len(locations) > 1:
            dupe_defs += (len(locations) - 1)
            for start, end in locations[:-1]:
                for i in range(start, end + 1):
                    lines_to_remove.add(i)

    # Process standard imports
    dupe_imports = 0
    for sig, locations in dedup_visitor.import_locations.items():
        if len(locations) > 1:
            dupe_imports += (len(locations) - 1)
            for start, end in locations[1:]:
                for i in range(start, end + 1):
                    lines_to_remove.add(i)

    print(f"Found {dupe_defs} duplicate definitions and {dupe_imports} redundant imports.")

    # 4. Reconstruct Source with Injections
    new_source_parts = []
    # enumerate is 1-based, matching AST lineno
    for i, line in enumerate(original_lines, 1):
        if i not in lines_to_remove:
            new_source_parts.append(line)
        
        # Check for injections needed AFTER this line
        # (Used for appending methods to the end of a class)
        if i in injections:
            for chunk in injections[i]:
                new_source_parts.append(chunk)

    intermediate_source = "".join(new_source_parts)

    # 5. Whitespace Cleanup
    print("Cleaning whitespace...")
    clean_source = clean_whitespace(intermediate_source)

    # 6. Auto-Healing / Verification Loop
    print("Verifying syntax and checking for floating decorators...")
    try:
        final_source = repair_source_code(clean_source)
    except SyntaxError as e:
        print("Failed to generate valid code. Original file restored.", file=sys.stderr)
        sys.exit(1)

    # 7. Write Output
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
