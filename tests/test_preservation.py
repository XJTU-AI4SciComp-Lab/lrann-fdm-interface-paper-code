"""Check all original executable syntax after expanding shared helper imports."""

import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class OriginalSingleLayerSyntax(ast.NodeTransformer):
    """Remove only the explicit optional sine branch; defaults retain the original."""

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if node.name == "main" and node.args.args[-1].arg == "outer_activation":
            assert ast.literal_eval(node.args.defaults[-1]) == "tanh"
            node.args.args.pop()
            node.args.defaults.pop()
        return node

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
            call = node.value
            if call.func.attr == "add_argument" and call.args and isinstance(call.args[0], ast.Constant) and call.args[0].value == "--outer-activation":
                return None
        return self.generic_visit(node)

    def visit_If(self, node):
        if ast.unparse(node.test) == "outer_activation == 'sin'":
            assert len(node.body) == 1 and ast.unparse(node.body[0]) == "model2.phi = torch.sin" and not node.orelse
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "main":
            node.keywords = [keyword for keyword in node.keywords if keyword.arg != "outer_activation"]
        return node


def expanded_tree(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "pathlib":
            if any(alias.asname == "_Path" for alias in node.names):
                continue
        if isinstance(node, ast.Import) and any(alias.asname == "_sys" for alias in node.names):
            continue
        if isinstance(node, ast.Import) and any(alias.name == "lrann_interface.compat" for alias in node.names):
            continue
        if isinstance(node, ast.Expr) and ast.unparse(node).startswith("_sys.path.insert("):
            continue
        if isinstance(node, ast.ImportFrom) and node.module.startswith("lrann_interface."):
            module = ROOT / "src" / (node.module.replace(".", "/") + ".py")
            functions = {f.name: f for f in ast.parse(module.read_text(encoding="utf-8")).body if isinstance(f, ast.FunctionDef)}
            for alias in node.names:
                function = copy.deepcopy(functions[alias.name])
                function.name = alias.asname or alias.name
                body.append(function)
        else:
            body.append(node)
    tree.body = body
    if path.name == "circle_2d_1l.py":
        tree = OriginalSingleLayerSyntax().visit(tree)
    return tree


def test_all_original_expressions_preserved():
    manifest = json.loads((ROOT / "tests/fixtures/original_ast.json").read_text())
    assert len(manifest) == 15
    assert set(manifest) == {p.relative_to(ROOT).as_posix() for p in (ROOT / "experiments").rglob("*.py")}
    for relative, record in manifest.items():
        digest = hashlib.sha256(ast.dump(expanded_tree(ROOT / relative), include_attributes=False).encode()).hexdigest()
        assert digest == record["sha256"], relative
