"""Import-boundary check (PLAN.md §2.1): static scan of every module's imports."""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).parents[1] / "src"
AREAS = ("metanoia_contracts", "metanoia", "metanoia_research", "metanoia_analysis")
ALLOWED = {
    "metanoia_contracts": {"metanoia_contracts"},
    "metanoia": {"metanoia", "metanoia_contracts"},
    "metanoia_research": {"metanoia_research", "metanoia", "metanoia_contracts"},
    "metanoia_analysis": {"metanoia_analysis", "metanoia_contracts"},
}
# Third-party packages an area must never import, even lazily.
FORBIDDEN_THIRD_PARTY = {
    "metanoia_contracts": {"torch", "numpy", "scipy", "pyarrow", "transformers", "faiss"},
    "metanoia_analysis": {"torch", "transformers", "faiss"},
}


def imported_roots(path: Path) -> set[str]:
    roots = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("area", AREAS)
def test_dependency_direction(area: str) -> None:
    for path in (SRC / area).rglob("*.py"):
        roots = imported_roots(path)
        project = roots & set(AREAS)
        assert project <= ALLOWED[area], f"{path} imports {project - ALLOWED[area]}"
        forbidden = roots & FORBIDDEN_THIRD_PARTY.get(area, set())
        assert not forbidden, f"{path} imports {forbidden}"


def test_core_top_level_import_is_lightweight() -> None:
    import subprocess
    import sys

    code = "import sys, metanoia, metanoia_contracts; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
