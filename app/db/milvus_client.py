from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.core.config import ROOT_DIR, get_settings

logger = logging.getLogger("app.milvus")


class MilvusStore:
    """Milvus store. Default: local Milvus Lite file via MilvusClient(uri=*.db)."""

    def __init__(self) -> None:
        settings = get_settings()
        self.collection_name = settings.milvus_collection
        self.dim = settings.embedding_dim
        self.mode = (settings.milvus_mode or "lite").lower()
        self.lite_path = settings.milvus_lite_path
        self.host = settings.milvus_host
        self.port = settings.milvus_port
        self._client = None
        self._memory: list[dict[str, Any]] = []
        self._use_memory = True
        self._init()

    def _resolve_uri(self) -> str:
        if self.mode == "server":
            return f"http://{self.host}:{self.port}"
        path = Path(self.lite_path or "data/milvus_demo.db")
        if not path.is_absolute():
            path = ROOT_DIR / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path.resolve())

    def _collection_dim(self) -> int | None:
        if self._client is None:
            return None
        try:
            info = self._client.describe_collection(self.collection_name)
            fields = info.get("fields") or info.get("schema", {}).get("fields") or []
            for field in fields:
                name = field.get("name") or field.get("field_name")
                if name != "embedding":
                    continue
                params = field.get("params") or field.get("type_params") or {}
                dim = params.get("dim") or field.get("dim")
                if dim is not None:
                    return int(dim)
        except Exception:
            logger.debug("describe_collection failed", exc_info=True)
        return None

    def _create_collection(self) -> None:
        from pymilvus import DataType, MilvusClient

        assert self._client is not None
        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("project_id", DataType.VARCHAR, max_length=64)
        schema.add_field("symbol_id", DataType.VARCHAR, max_length=64)
        schema.add_field("fqcn", DataType.VARCHAR, max_length=512)
        schema.add_field("module_name", DataType.VARCHAR, max_length=128)
        schema.add_field("tags", DataType.VARCHAR, max_length=1024)
        schema.add_field("text", DataType.VARCHAR, max_length=4096)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dim)
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="IP",
        )
        self._client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        self._ensure_loaded()
        logger.info(
            "created milvus collection=%s dim=%s",
            self.collection_name,
            self.dim,
        )

    def _ensure_loaded(self) -> None:
        """Milvus server collections may be released; search requires load()."""
        if self._client is None or self._use_memory:
            return
        try:
            self._client.load_collection(self.collection_name)
        except Exception as e:
            # Lite / already-loaded often raise harmless errors.
            msg = str(e).lower()
            if "already loaded" in msg or "not found" in msg:
                return
            logger.debug("load_collection(%s): %s", self.collection_name, e)

    def recreate_collection(self, dim: int | None = None) -> None:
        """Drop and recreate collection (e.g. after embedding model/dim change)."""
        if dim is not None and dim > 0:
            self.dim = int(dim)
        if self._client is None:
            self._init()
            return
        try:
            if self._client.has_collection(self.collection_name):
                self._client.drop_collection(self.collection_name)
                logger.warning(
                    "dropped milvus collection=%s (recreate for dim=%s)",
                    self.collection_name,
                    self.dim,
                )
        except Exception:
            logger.exception("drop milvus collection failed")
        self._create_collection()
        self._use_memory = False

    def _ensure_collection_dim(self, dim: int) -> None:
        if dim <= 0:
            return
        self.dim = dim
        if self._client is None:
            return
        if not self._client.has_collection(self.collection_name):
            self._create_collection()
            return
        existing = self._collection_dim()
        if existing is not None and existing != dim:
            logger.warning(
                "milvus dim mismatch collection=%s expected=%s got=%s — recreating",
                self.collection_name,
                existing,
                dim,
            )
            self.recreate_collection(dim)

    def _init(self) -> None:
        # Avoid pymilvus reading our lite path from env as a server URI
        import os

        os.environ.pop("MILVUS_URI", None)
        try:
            from pymilvus import MilvusClient

            uri = self._resolve_uri()
            self._client = MilvusClient(uri=uri)
            if not self._client.has_collection(self.collection_name):
                self._create_collection()
            else:
                existing = self._collection_dim()
                if existing is not None and existing != self.dim:
                    logger.warning(
                        "milvus collection dim=%s != config dim=%s — recreating",
                        existing,
                        self.dim,
                    )
                    self.recreate_collection(self.dim)
                else:
                    self._ensure_loaded()
            self._use_memory = False
        except Exception:
            logger.exception("milvus init failed; falling back to in-memory store")
            self._client = None
            self._use_memory = True

    def verify(self) -> bool:
        return not self._use_memory and self._client is not None

    def upsert(
        self,
        rows: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        if not rows:
            return
        if embeddings:
            self._ensure_collection_dim(len(embeddings[0]))

        if self._use_memory or self._client is None:
            by_id = {r["id"]: {**r, "embedding": emb} for r, emb in zip(rows, embeddings)}
            existing = {r["id"]: r for r in self._memory}
            existing.update(by_id)
            self._memory = list(existing.values())
            return

        data = []
        for r, emb in zip(rows, embeddings):
            data.append(
                {
                    "id": r["id"],
                    "project_id": r["project_id"],
                    "symbol_id": r["symbol_id"],
                    "fqcn": (r.get("fqcn") or "")[:500],
                    "module_name": (r.get("module_name") or "")[:128],
                    "tags": (r.get("tags") or "")[:1000],
                    "text": (r.get("text") or "")[:4000],
                    "embedding": emb,
                }
            )
        try:
            self._client.upsert(collection_name=self.collection_name, data=data)
        except Exception as e:
            msg = str(e).lower()
            if "expected dim" in msg or "dim" in msg:
                logger.warning("milvus upsert dim error, recreating collection: %s", e)
                self.recreate_collection(len(embeddings[0]) if embeddings else self.dim)
                self._client.upsert(collection_name=self.collection_name, data=data)
            else:
                raise

    def search(
        self,
        query_embedding: list[float],
        *,
        project_id: str,
        top_k: int = 12,
    ) -> list[dict[str, Any]]:
        if query_embedding:
            self._ensure_collection_dim(len(query_embedding))

        if self._use_memory or self._client is None:
            scored: list[tuple[float, dict[str, Any]]] = []
            for row in self._memory:
                if row.get("project_id") != project_id:
                    continue
                emb = row.get("embedding") or []
                score = sum(a * b for a, b in zip(query_embedding, emb))
                scored.append((score, row))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [
                {
                    "symbol_id": r["symbol_id"],
                    "fqcn": r["fqcn"],
                    "module_name": r.get("module_name", ""),
                    "tags": r.get("tags", ""),
                    "text": r.get("text", ""),
                    "score": float(s),
                }
                for s, r in scored[:top_k]
            ]

        self._ensure_loaded()
        try:
            results = self._client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=top_k,
                filter=f'project_id == "{project_id}"',
                output_fields=["symbol_id", "fqcn", "module_name", "tags", "text", "project_id"],
                search_params={"metric_type": "IP"},
            )
        except Exception as e:
            msg = str(e).lower()
            if "released" in msg or "load()" in msg or "not loaded" in msg:
                logger.warning("milvus collection released; reloading and retrying search")
                self._ensure_loaded()
                results = self._client.search(
                    collection_name=self.collection_name,
                    data=[query_embedding],
                    limit=top_k,
                    filter=f'project_id == "{project_id}"',
                    output_fields=["symbol_id", "fqcn", "module_name", "tags", "text", "project_id"],
                    search_params={"metric_type": "IP"},
                )
            else:
                raise
        hits: list[dict[str, Any]] = []
        for hit in results[0]:
            entity = hit.get("entity") or hit
            hits.append(
                {
                    "symbol_id": entity.get("symbol_id"),
                    "fqcn": entity.get("fqcn"),
                    "module_name": entity.get("module_name"),
                    "tags": entity.get("tags"),
                    "text": entity.get("text"),
                    "score": float(hit.get("distance", hit.get("score", 0.0))),
                }
            )
        return hits

    def clear_project(self, project_id: str) -> None:
        if self._use_memory or self._client is None:
            self._memory = [r for r in self._memory if r.get("project_id") != project_id]
            return
        try:
            self._client.delete(
                collection_name=self.collection_name,
                filter=f'project_id == "{project_id}"',
            )
        except Exception:
            pass


_milvus: MilvusStore | None = None
_entry_route: "EntryRouteStore | None" = None


def get_milvus() -> MilvusStore:
    global _milvus
    if _milvus is None:
        _milvus = MilvusStore()
    return _milvus


def reset_milvus() -> MilvusStore:
    """Force re-init (e.g. after dim/config change)."""
    global _milvus, _entry_route
    _milvus = None
    _entry_route = None
    return get_milvus()


class EntryRouteStore:
    """Second Milvus collection for entry routing (Class#method anchors).

    Reuses the same MilvusClient as MilvusStore so Lite file locks stay single-process.
    """

    OUTPUT_FIELDS = [
        "symbol_id",
        "project_id",
        "kind",
        "class_method",
        "fqcn",
        "method_name",
        "http_paths",
        "mq_topic",
        "flow_role",
        "api_doc",
        "labels",
        "routing_text",
    ]

    def __init__(self, milvus: MilvusStore | None = None) -> None:
        settings = get_settings()
        self.collection_name = settings.milvus_entry_route_collection
        self._base = milvus or get_milvus()
        self.dim = self._base.dim
        self._ensure_collection()

    @property
    def _client(self):
        return self._base._client

    @property
    def _use_memory(self) -> bool:
        return bool(self._base._use_memory or self._client is None)

    def verify(self) -> bool:
        return self._base.verify() and not self._use_memory

    def _collection_dim(self) -> int | None:
        if self._client is None:
            return None
        try:
            info = self._client.describe_collection(self.collection_name)
            fields = info.get("fields") or info.get("schema", {}).get("fields") or []
            for field in fields:
                name = field.get("name") or field.get("field_name")
                if name != "embedding":
                    continue
                params = field.get("params") or field.get("type_params") or {}
                dim = params.get("dim") or field.get("dim")
                if dim is not None:
                    return int(dim)
        except Exception:
            logger.debug("entry_route describe_collection failed", exc_info=True)
        return None

    def _create_collection(self) -> None:
        from pymilvus import DataType, MilvusClient

        assert self._client is not None
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("project_id", DataType.VARCHAR, max_length=64)
        schema.add_field("symbol_id", DataType.VARCHAR, max_length=64)
        schema.add_field("kind", DataType.VARCHAR, max_length=64)
        schema.add_field("class_method", DataType.VARCHAR, max_length=256)
        schema.add_field("fqcn", DataType.VARCHAR, max_length=512)
        schema.add_field("method_name", DataType.VARCHAR, max_length=256)
        schema.add_field("http_paths", DataType.VARCHAR, max_length=1024)
        schema.add_field("mq_topic", DataType.VARCHAR, max_length=512)
        schema.add_field("flow_role", DataType.VARCHAR, max_length=32)
        schema.add_field("api_doc", DataType.VARCHAR, max_length=512)
        schema.add_field("labels", DataType.VARCHAR, max_length=512)
        schema.add_field("routing_text", DataType.VARCHAR, max_length=2048)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dim)
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="IP",
        )
        self._client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        self._ensure_loaded()
        logger.info(
            "created milvus entry_route collection=%s dim=%s",
            self.collection_name,
            self.dim,
        )

    def _ensure_loaded(self) -> None:
        if self._client is None or self._use_memory:
            return
        try:
            self._client.load_collection(self.collection_name)
        except Exception as e:
            msg = str(e).lower()
            if "already loaded" in msg or "not found" in msg:
                return
            logger.debug("load_collection(%s): %s", self.collection_name, e)

    def _ensure_collection(self) -> None:
        if self._client is None:
            return
        try:
            if not self._client.has_collection(self.collection_name):
                self.dim = self._base.dim
                self._create_collection()
                return
            existing = self._collection_dim()
            if existing is not None and existing != self._base.dim and self._base.dim > 0:
                logger.warning(
                    "entry_route dim mismatch collection=%s expected=%s got=%s — recreating",
                    self.collection_name,
                    self._base.dim,
                    existing,
                )
                self.recreate_collection(self._base.dim)
            else:
                if existing:
                    self.dim = existing
                self._ensure_loaded()
        except Exception:
            logger.exception("entry_route ensure_collection failed")

    def recreate_collection(self, dim: int | None = None) -> None:
        if dim is not None and dim > 0:
            self.dim = int(dim)
        if self._client is None:
            return
        try:
            if self._client.has_collection(self.collection_name):
                self._client.drop_collection(self.collection_name)
                logger.warning(
                    "dropped entry_route collection=%s (recreate dim=%s)",
                    self.collection_name,
                    self.dim,
                )
        except Exception:
            logger.exception("drop entry_route collection failed")
        self._create_collection()

    def clear_project(self, project_id: str) -> None:
        if self._use_memory or self._client is None:
            return
        try:
            self._client.delete(
                collection_name=self.collection_name,
                filter=f'project_id == "{project_id}"',
            )
        except Exception:
            logger.debug("entry_route clear_project failed", exc_info=True)

    def upsert(self, rows: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
        if not rows:
            return
        if embeddings and len(embeddings[0]) > 0:
            self.dim = len(embeddings[0])
            self._base.dim = self.dim
            existing = self._collection_dim()
            if existing is not None and existing != self.dim:
                self.recreate_collection(self.dim)
            elif self._client and not self._client.has_collection(self.collection_name):
                self._create_collection()

        if self._use_memory or self._client is None:
            logger.warning("entry_route upsert skipped (memory/unavailable)")
            return

        data = []
        for r, emb in zip(rows, embeddings):
            data.append(
                {
                    "id": str(r.get("id") or r.get("symbol_id") or "")[:64],
                    "project_id": str(r.get("project_id") or "")[:64],
                    "symbol_id": str(r.get("symbol_id") or "")[:64],
                    "kind": str(r.get("kind") or "")[:64],
                    "class_method": str(r.get("class_method") or "")[:256],
                    "fqcn": str(r.get("fqcn") or "")[:500],
                    "method_name": str(r.get("method_name") or "")[:256],
                    "http_paths": str(r.get("http_paths") or "")[:1000],
                    "mq_topic": str(r.get("mq_topic") or "")[:500],
                    "flow_role": str(r.get("flow_role") or "main")[:32],
                    "api_doc": str(r.get("api_doc") or "")[:500],
                    "labels": str(r.get("labels") or "")[:500],
                    "routing_text": str(r.get("routing_text") or "")[:2000],
                    "embedding": emb,
                }
            )
        self._client.upsert(collection_name=self.collection_name, data=data)

    def search(
        self,
        query_embedding: list[float],
        *,
        project_id: str,
        top_k: int = 8,
    ) -> list[dict[str, Any]]:
        if self._use_memory or self._client is None:
            return []
        if query_embedding:
            self.dim = len(query_embedding)
        self._ensure_loaded()
        try:
            results = self._client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=top_k,
                filter=f'project_id == "{project_id}"',
                output_fields=list(self.OUTPUT_FIELDS),
                search_params={"metric_type": "IP"},
            )
        except Exception as e:
            msg = str(e).lower()
            if "not exist" in msg or "not found" in msg:
                logger.warning("entry_route collection missing; creating empty")
                self._create_collection()
                return []
            if "released" in msg or "load()" in msg or "not loaded" in msg:
                self._ensure_loaded()
                results = self._client.search(
                    collection_name=self.collection_name,
                    data=[query_embedding],
                    limit=top_k,
                    filter=f'project_id == "{project_id}"',
                    output_fields=list(self.OUTPUT_FIELDS),
                    search_params={"metric_type": "IP"},
                )
            else:
                raise
        hits: list[dict[str, Any]] = []
        for hit in results[0]:
            entity = hit.get("entity") or hit
            row = {k: entity.get(k) for k in self.OUTPUT_FIELDS}
            row["score"] = float(hit.get("distance", hit.get("score", 0.0)))
            hits.append(row)
        return hits


def get_entry_route_store() -> EntryRouteStore:
    global _entry_route
    if _entry_route is None:
        _entry_route = EntryRouteStore(get_milvus())
    return _entry_route
