"""Offline analysis: overview → data-source entries → recursive flows (+ inline entities).

Also: global Tree-sitter AST + topological priority AI (``run_topo_ast_pipeline``).
"""

from app.analysis.discover import discover_and_index_entries, describe_entity
from app.analysis.overview import build_project_overview
from app.analysis.recursive_flow import recursive_flow_from_entries
from app.analysis.topo_pipeline import run_topo_ast_pipeline

__all__ = [
    "build_project_overview",
    "discover_and_index_entries",
    "describe_entity",
    "recursive_flow_from_entries",
    "run_topo_ast_pipeline",
]
