from fastapi import APIRouter

from app.api import (
    agent_remote,
    chat,
    connectors,
    graph,
    health,
    ingest,
    knowledge,
    logs,
    projects,
    search,
)

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(projects.router)
api_router.include_router(ingest.router)
api_router.include_router(search.router)
api_router.include_router(graph.router)
api_router.include_router(chat.router)
api_router.include_router(agent_remote.router)
api_router.include_router(knowledge.router)
api_router.include_router(connectors.router)
api_router.include_router(logs.router)
