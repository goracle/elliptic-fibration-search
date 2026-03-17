#!/usr/bin/env python3
import ast, argparse, os, re, shutil, sys, tempfile, textwrap
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

"""
dedup.py - Advanced Python deduplication with Class Merging, AST parsing, and Auto-Healing.

Features:
- Merges duplicate top-level Class definitions (LAST = master).
- Moves unique methods from earlier class defs into the master.
- Removes duplicate global function definitions (keeps LAST).
- Normalizes, deduplicates, and hoists global imports to the top,
  but preserves grouping (comma-separated) and first-seen order.
- Removes duplicate top-level comment blocks (identical consecutive comment lines).
- Cleans excessive vertical whitespace (max 2 blank lines).
- Auto-heals orphaned decorators causing SyntaxError by removing decorator lines.
- Preserves file permissions and creates backups.
- Conservative: raises on unexpected AST limitations.
"""

# -------------------------
# Exceptions & Utilities
# -------------------------

class DedupError(Exception):
    pass

def read_file_lines(filename: str) -> List[str]:
    with open(filename, "r", encoding="utf-8") as f:
        return f.read().splitlines(keepends=True)

def write_atomic(filename: str, content: str) -> None:
    dirpath = os.path.dirname(filename) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dirpath, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        shutil.copymode(filename, tmp_path)
        shutil.move(tmp_path, filename)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

def ensure_end_lineno_support(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if hasattr(node, "lineno") and not hasattr(node, "end_lineno"):
            raise DedupError("AST nodes missing end_lineno; Python >= 3.8 required.")

# -------------------------
# Class merging pass
# -------------------------

class ClassInfo:
    def __init__(self, node: ast.ClassDef):
        self.node = node
        self.name = node.name
        self.start = 0
        self.end = 0
        self.methods: Dict[str, ast.AST] = {}
        self.method_nodes: Dict[str, ast.AST] = {}

    def analyze(self) -> None:
        self.start = min([self.node.lineno] + [d.lineno for d in getattr(self.node, "decorator_list", [])] or [self.node.lineno])
        self.end = getattr(self.node, "end_lineno", self.node.lineno)
        for child in self.node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.methods[child.name] = child
                self.method_nodes[child.name] = child

# -------------------------
# Standard dedup pass
# -------------------------

# -------------------------
# Import normalization & hoisting (GROUPED)
# -------------------------

# -------------------------
# Top-level comment block dedupe
# -------------------------

def remove_duplicate_top_level_comment_blocks(lines: List[str], tree: ast.AST) -> List[str]:
    func_ranges: List[Tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_ranges.append((node.lineno, getattr(node, "end_lineno", node.lineno)))
    def inside_func(ln: int) -> bool:
        return any(s <= ln <= e for s, e in func_ranges)
    out: List[str] = []
    seen_blocks: Set[Tuple[str, ...]] = set()
    i = 0
    n = len(lines)
    while i < n:
        ln_no = i + 1
        line = lines[i]
        if line.lstrip().startswith("#") and not inside_func(ln_no):
            block = [line]
            j = i + 1
            while j < n and lines[j].lstrip().startswith("#") and not inside_func(j + 1):
                block.append(lines[j])
                j += 1
            key = tuple(block)
            if key not in seen_blocks:
                seen_blocks.add(key)
                out.extend(block)
            i = j
        else:
            out.append(line)
            i += 1
    return out

# -------------------------
# Whitespace cleanup
# -------------------------

def clean_whitespace(source: str) -> str:
    source = re.sub(r"\n{3,}", "\n\n", source)
    source = re.sub(r"[ \t]+\n", "\n", source)
    return source

# -------------------------
# Repair / auto-heal
# -------------------------

def repair_source_code(source: str, max_retries: int = 40) -> str:
    current = source
    for attempt in range(max_retries):
        try:
            ast.parse(current)
            return current
        except SyntaxError as e:
            if e.lineno is None:
                raise
            lines = current.splitlines(True)
            idx = e.lineno - 1
            if 0 <= idx < len(lines):
                err_line = lines[idx]
                if err_line.strip().startswith("@"):
                    del lines[idx]
                    current = "".join(lines)
                    continue
            raise
    raise DedupError("repair_source_code: exceeded max retries")

# -------------------------
# Main processing
# -------------------------

def process_file(filename: str) -> None:
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    bak = f"{filename}.bak"
    shutil.copy2(filename, bak)
    print(f"[backup] created: {bak}")

    original_lines = read_file_lines(filename)
    source = "".join(original_lines)

    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as e:
        raise SyntaxError(f"Original file has syntax errors. Aborting: {e}")

    ensure_end_lineno_support(tree)

    # Class merging
    merger = ClassMergeAnalyzer(original_lines)
    merger.visit(tree)
    merged_names = merger.process_merges()
    lines_to_remove: Set[int] = set(merger.lines_to_remove)
    injections: Dict[int, List[str]] = defaultdict(list)
    for k, v in merger.injections.items():
        injections[k].extend(v)
    if merged_names:
        print(f"[class-merge] merged classes: {', '.join(sorted(merged_names))}")

    # Standard dedup visitor
    dedup = StandardDedupVisitor(ignore_classes=merged_names)
    dedup.visit(tree)

    # Remove duplicate global defs (keep last)
    for name, locs in dedup.definition_locations.items():
        if len(locs) > 1:
            for start, end in locs[:-1]:
                for ln in range(start, end + 1):
                    lines_to_remove.add(ln)

    # Mark redundant import lines (beyond first) for removal — normalization will rebuild grouped imports
    for sig, locs in dedup.import_locations.items():
        if len(locs) > 1:
            for start, end in locs[1:]:
                for ln in range(start, end + 1):
                    lines_to_remove.add(ln)

    # Reconstruct with removals and injections
    new_parts: List[str] = []
    for i, line in enumerate(original_lines, start=1):
        if i not in lines_to_remove:
            new_parts.append(line)
        if i in injections:
            for chunk in injections[i]:
                new_parts.append(chunk)

    intermediate_lines = new_parts
    intermediate_source = "".join(intermediate_lines)

    # Parse intermediate and try auto-heal if needed
    try:
        new_tree = ast.parse(intermediate_source, filename=filename)
    except SyntaxError:
        print("[repair] intermediate parse failed; attempting auto-heal...", file=sys.stderr)
        intermediate_source = repair_source_code(intermediate_source)
        new_tree = ast.parse(intermediate_source, filename=filename)

    ensure_end_lineno_support(new_tree)

    # Import normalization (grouped) and hoisting
    normalized_lines, import_line_numbers = normalize_and_hoist_imports_grouped(intermediate_source.splitlines(keepends=True), new_tree)
    normalized_source = "".join(normalized_lines)

    # Parse normalized, auto-heal if necessary
    try:
        tree_after_imports = ast.parse(normalized_source, filename=filename)
    except SyntaxError:
        normalized_source = repair_source_code(normalized_source)
        tree_after_imports = ast.parse(normalized_source, filename=filename)

    # Remove duplicate top-level comment blocks
    final_lines = remove_duplicate_top_level_comment_blocks(normalized_source.splitlines(keepends=True), tree_after_imports)
    final_source = "".join(final_lines)

    # Whitespace cleanup
    final_source = clean_whitespace(final_source)

    # Final verification and auto-heal
    try:
        final_source = repair_source_code(final_source)
    except SyntaxError as e:
        shutil.copy2(bak, filename)
        raise SyntaxError(f"Final verification failed after auto-heal. Original restored from {bak}. Error: {e}")

    write_atomic(filename, final_source)
    print(f"[success] updated: {filename}")

# -------------------------
# CLI
# -------------------------

def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Deduplicate Python code, merge classes, normalize imports (grouped), and auto-heal.")
    parser.add_argument("filename", help="Python file to process")
    args = parser.parse_args(argv)

    try:
        process_file(args.filename)
    except Exception:
        print(f"[fatal] processing failed for {args.filename}", file=sys.stderr)
        raise

class ClassMergeAnalyzer(ast.NodeVisitor):
    """
    Analyzes classes for merging.
    Now supports nested classes and deep method deduplication.
    """
    def __init__(self, source_lines: List[str]):
        self.source_lines = source_lines
        self.classes: Dict[str, List[ClassInfo]] = defaultdict(list)
        self.lines_to_remove: Set[int] = set()
        self.injections: Dict[int, List[str]] = defaultdict(list)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # REMOVED: col_offset == 0 check.
        # This allows merging even if the class is indented or nested.
        info = ClassInfo(node)
        info.analyze()
        self.classes[node.name].append(info)
        self.generic_visit(node)

    def process_merges(self) -> Set[str]:
        processed: Set[str] = set()
        for name, infos in list(self.classes.items()):
            if len(infos) < 2:
                continue

            processed.add(name)
            # The LAST definition in the file is the "Master"
            master = infos[-1]

            # Determine indentation from master class body
            master_indent = "    "
            if master.node.body:
                first_node = master.node.body[0]
                master_indent = " " * getattr(first_node, "col_offset", 4)

            for prev in infos[:-1]:
                # 1. Mark the entire old class block for removal
                for ln in range(prev.start, prev.end + 1):
                    self.lines_to_remove.add(ln)

                # 2. Deep Method Merging
                # If 'prev' has a method that 'master' doesn't, we hoist it.
                # If 'master' already has it, we assume 'master' (being later)
                # has the updated version and we discard the 'prev' version.
                for mname, mnode in prev.methods.items():
                    if mname not in master.methods:
                        # Extract raw text for the method (including decorators)
                        m_start = min([mnode.lineno] + [d.lineno for d in getattr(mnode, "decorator_list", [])] or [mnode.lineno])
                        m_end = getattr(mnode, "end_lineno", mnode.lineno)

                        raw = "".join(self.source_lines[m_start - 1 : m_end])
                        dedented = textwrap.dedent(raw)

                        # Re-indent to match the master class's internal indentation
                        reindented = "".join(
                            (master_indent + line) if line.strip() else line
                            for line in dedented.splitlines(True)
                        )

                        # Inject into the end of the master class
                        chunk = "\n" + reindented.rstrip() + "\n"
                        self.injections[master.end].append(chunk)

                        # Update master's method map so we don't double-inject
                        master.methods[mname] = mnode

        return processed

class StandardDedupVisitor(ast.NodeVisitor):
    """
    Scans for duplicate global functions and imports.
    Updated to be more resilient to indentation and scope.
    """
    def __init__(self, ignore_classes: Set[str]):
        self.ignore_classes = ignore_classes
        # name -> list of (start_line, end_line)
        self.definition_locations: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        # signature -> list of (start_line, end_line)
        self.import_locations: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.scope_stack: List[str] = []

    def _get_full_range(self, node: ast.AST) -> Tuple[int, int]:
        """Calculates line range including decorators."""
        start = node.lineno
        if hasattr(node, "decorator_list") and node.decorator_list:
            start = min(d.lineno for d in node.decorator_list)
        end = getattr(node, "end_lineno", node.lineno)
        return start, end

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # If the class was already handled by ClassMergeAnalyzer,
        # we don't treat it as a 'global definition' to be deleted here.
        if node.name in self.ignore_classes:
            self.scope_stack.append(node.name)
            self.generic_visit(node)
            self.scope_stack.pop()
            return

        # Otherwise, track it for potential total-replacement deduplication
        if not self.scope_stack:
            self.definition_locations[node.name].append(self._get_full_range(node))

        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # We only deduplicate top-level functions.
        # Methods inside classes are handled by ClassMergeAnalyzer.
        if not self.scope_stack:
            start, end = self._get_full_range(node)
            self.definition_locations[node.name].append((start, end))
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_Import(self, node: ast.Import) -> None:
        if not self.scope_stack:
            sig = "import " + ", ".join(sorted(a.name for a in node.names))
            self.import_locations[sig].append((node.lineno, getattr(node, "end_lineno", node.lineno)))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if not self.scope_stack:
            module = node.module or ""
            names = ", ".join(sorted(a.name for a in node.names))
            level = "." * (node.level or 0)
            sig = f"from {level}{module} import {names}"
            self.import_locations[sig].append((node.lineno, getattr(node, "end_lineno", node.lineno)))

    def _start_line(self, node: ast.AST) -> int:
        if hasattr(node, "lineno"):
            start = node.lineno
            if hasattr(node, "decorator_list") and getattr(node, "decorator_list"):
                start = min(d.lineno for d in node.decorator_list)
            return start
        raise DedupError("AST node missing lineno in _start_line")

def normalize_and_hoist_imports_grouped(source_lines: List[str], tree: ast.AST) -> Tuple[List[str], Set[int]]:
    imports_nodes: List[ast.AST] = []
    import_line_numbers: Set[int] = set()

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports_nodes.append(node)
            for ln in range(node.lineno, getattr(node, "end_lineno", node.lineno) + 1):
                import_line_numbers.add(ln)

    future_imports: List[str] = []
    top_imports_order: List[Tuple[str, str]] = []
    top_seen: Set[Tuple[str, str]] = set()
    from_imports_order: List[Tuple[Tuple[int, str], List[Tuple[str, str]]]] = []
    from_seen: Dict[Tuple[int, str], Set[Tuple[str, str]]] = {}
    from_has_star: Dict[Tuple[int, str], bool] = {}

    for node in imports_nodes:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            # Special case: __future__ must be at the very top
            names = ", ".join(sorted(a.name for a in node.names))
            future_imports.append(f"from __future__ import {names}")
            continue

        if isinstance(node, ast.Import):
            for alias in node.names:
                key = (alias.name, alias.asname)
                if key not in top_seen:
                    top_seen.add(key)
                    top_imports_order.append(key)
        else: # ast.ImportFrom
            module = node.module or ""
            level = getattr(node, "level", 0) or 0
            mod_key = (level, module)
            if mod_key not in from_seen:
                from_seen[mod_key] = set()
                from_imports_order.append((mod_key, []))
                from_has_star[mod_key] = False

            for alias in node.names:
                if alias.name == "*":
                    from_has_star[mod_key] = True
                    from_seen[mod_key].clear()
                    for (mk, lst) in from_imports_order:
                        if mk == mod_key:
                            lst.clear()
                            break
                    continue

                if from_has_star.get(mod_key, False):
                    continue

                akey = (alias.name, alias.asname)
                if akey not in from_seen[mod_key]:
                    from_seen[mod_key].add(akey)
                    for (mk, lst) in from_imports_order:
                        if mk == mod_key:
                            lst.append(akey)
                            break

    normalized_lines: List[str] = []

    # 1. ALWAYS LEAD WITH FUTURE
    normalized_lines.extend(future_imports)

    # 2. Standard imports
    if top_imports_order:
        parts = [f"{n} as {a}" if a else n for n, a in top_imports_order]
        normalized_lines.append(f"import {', '.join(parts)}")

    # 3. From imports
    for (level, module), aliases in from_imports_order:
        lvl = "." * level if level else ""
        if from_has_star.get((level, module)):
            normalized_lines.append(f"from {lvl}{module} import *")
        elif aliases:
            parts = [f"{n} as {a}" if a else n for n, a in aliases]
            normalized_lines.append(f"from {lvl}{module} import {', '.join(parts)}")

    remaining = [line for i, line in enumerate(source_lines, start=1) if i not in import_line_numbers]

    # Assemble with shebang preservation
    preserved_prefix = []
    if remaining and (remaining[0].startswith("#!") or "coding" in remaining[0]):
        preserved_prefix.append(remaining.pop(0))

    final_imports = [ln if ln.endswith("\n") else ln + "\n" for ln in normalized_lines]
    return preserved_prefix + final_imports + ["\n"] + remaining, import_line_numbers

if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise
