"""Stable provenance fingerprints for generated teaching data."""

import ast
import hashlib
from pathlib import Path


IMPLEMENTATION_HASH_KIND = "python-ast-v1"
_GENERATION_ROOTS = ("rebuild_meta_dataset",)
_IGNORED_AST_FIELDS = {"type_comment", "type_ignores", "type_params"}


class _StripDocstrings(ast.NodeTransformer):
    """Remove documentation that cannot affect generated values."""

    def _visit_body(self, node):
        self.generic_visit(node)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body.pop(0)
        return node

    visit_Module = _visit_body
    visit_FunctionDef = _visit_body
    visit_AsyncFunctionDef = _visit_body
    visit_ClassDef = _visit_body


def _bound_names(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = set()
        pending = list(targets)
        while pending:
            target = pending.pop()
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, (ast.List, ast.Tuple)):
                pending.extend(target.elts)
        return names
    return set()


def _import_bindings(node):
    if isinstance(node, ast.Import):
        return {
            alias.asname
            or alias.name.split(".")[0]: (
                "import",
                alias.name,
                alias.asname,
            )
            for alias in node.names
        }
    if isinstance(node, ast.ImportFrom):
        return {
            alias.asname
            or alias.name: (
                "from",
                node.level,
                node.module,
                alias.name,
                alias.asname,
            )
            for alias in node.names
        }
    return {}


def _canonical_ast(value):
    """Return a Python-version-neutral representation of executable syntax."""
    if isinstance(value, ast.AST):
        return (
            type(value).__name__,
            tuple(
                (name, _canonical_ast(child))
                for name, child in ast.iter_fields(value)
                if name not in _IGNORED_AST_FIELDS
            ),
        )
    if isinstance(value, list):
        return tuple(_canonical_ast(child) for child in value)
    return value


def implementation_sha256(path):
    """Hash the normalized AST that can contribute to the generated meta-dataset."""
    path = Path(path)
    module = ast.parse(path.read_text(), filename=str(path))
    symbols = {name: node for node in module.body for name in _bound_names(node)}
    imports = {
        name: binding
        for node in module.body
        for name, binding in _import_bindings(node).items()
    }
    missing = set(_GENERATION_ROOTS) - symbols.keys()
    if missing:
        raise ValueError(
            f"Missing generation entry point: {', '.join(sorted(missing))}"
        )

    selected = set()
    pending = list(_GENERATION_ROOTS)
    while pending:
        name = pending.pop()
        node = symbols[name]
        if id(node) in selected:
            continue
        selected.add(id(node))
        referenced = {
            child.id
            for child in ast.walk(node)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
        }
        pending.extend(referenced & symbols.keys())

    normalized = ast.Module(
        body=[node for node in module.body if id(node) in selected],
        type_ignores=[],
    )
    normalized = _StripDocstrings().visit(normalized)
    referenced = {
        child.id
        for child in ast.walk(normalized)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }
    relevant_imports = tuple(
        sorted((name, imports[name]) for name in referenced & imports.keys())
    )
    canonical = repr((relevant_imports, _canonical_ast(normalized)))
    payload = f"{IMPLEMENTATION_HASH_KIND}\0{canonical}".encode()
    return hashlib.sha256(payload).hexdigest()
