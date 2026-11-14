import ast, io, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1] / "app"
BANNED = ("放款","提现","授信","用信","额度","融担","机构限额","门店限额","优惠券","贴息","会员价格","二级门店","风控","担保")
def docstrings(tree):
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                ids.add(id(body[0].value))
    return ids
def code_strings(tree):
    skip = docstrings(tree)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in skip:
                out.append(node.value)
    return out
for p in ROOT.rglob("*.py"):
    rel = p.relative_to(ROOT.parent).as_posix()
    if rel in {"app/analysis/interface_tables.py", "app/analysis/table_dict.py"}:
        continue
    try:
        tree = ast.parse(p.read_text(encoding="utf-8"))
    except SyntaxError:
        continue
    for v in code_strings(tree):
        for noun in BANNED:
            if noun in v:
                print(f"{rel} :: {noun!r} in {v[:100]!r}")
