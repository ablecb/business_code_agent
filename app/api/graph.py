from __future__ import annotations

from fastapi import APIRouter, Query

from app.core.config import get_settings
from app.graph.builder import query_flow
from app.models.schemas import GraphEdge, GraphFlowResponse, GraphNode

router = APIRouter(tags=["graph"])


@router.get("/graph/flow", response_model=GraphFlowResponse)
def graph_flow(
    entry: str = Query(..., min_length=1),
    project_id: str | None = None,
    depth: int | None = None,
) -> GraphFlowResponse:
    settings = get_settings()
    data = query_flow(
        entry,
        project_id or settings.project_id,
        depth=depth or settings.graph_max_depth,
    )
    nodes = [GraphNode(**n) for n in data.get("nodes") or []]
    edges = [GraphEdge(**e) for e in data.get("edges") or []]
    return GraphFlowResponse(entry=data.get("entry") or entry, nodes=nodes, edges=edges)
