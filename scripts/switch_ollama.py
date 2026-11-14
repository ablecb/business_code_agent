#!/usr/bin/env python
"""One-click switch .env to intranet Ollama (qwen3 etc.).

Example:
    python scripts/switch_ollama.py                 # dry-run preview
    python scripts/switch_ollama.py --apply         # write .env
    python scripts/switch_ollama.py --apply --host http://192.0.2.100:11434 --model qwen3:32b
    python scripts/switch_ollama.py --restore       # restore latest backup

Notes:
    * qwen3 has NO embedding capability -> EMBEDDING_PROVIDER=openai_compatible
      pointing at Ollama's OpenAI-compatible /v1/embeddings (needs bge-m3 or
      nomic-embed-text pulled in Ollama first).
    * When embedding dimension changes (or first switch), delete the old
      Milvus lite file (data/milvus_demo.db) and re-run ingest, otherwise
      searches hit stale dimensions.
"""
from __future__ import annotations

import argparse
import datetime
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"

# key -> (default value, help)
OLLAMA_KEYS: dict[str, tuple[str, str]] = {
    "LLM_PROVIDER": ("ollama", "ollama provider is already supported by app/llm/factory.py"),
    "LLM_MODEL": ("qwen3:32b", "Ollama model tag (e.g. qwen3:32b, your GGUF tag)"),
    "LLM_BASE_URL": ("http://127.0.0.1:11434", "Ollama server base URL (no /v1 needed)"),
    "LLM_API_KEY": ("ollama", "Ollama ignores it; keep non-empty for compatibility"),
    "EMBEDDING_PROVIDER": ("openai_compatible", "qwen3 has no embeddings; use Ollama /v1/embeddings"),
    "EMBEDDING_MODEL": ("bge-m3", "pull it first: ollama pull bge-m3 (or nomic-embed-text)"),
    "EMBEDDING_BASE_URL": ("http://127.0.0.1:11434/v1", "Ollama OpenAI-compatible endpoint"),
    "EMBEDDING_API_KEY": ("ollama", "Ollama ignores it"),
    "EMBEDDING_DIM": ("1024", "bge-m3=1024, nomic-embed-text=768"),
    "MILVUS_MODE": ("lite", "lite=local file (no Docker); server needs compose stack"),
    # Qwen3 / long-context model tuning
    "LLM_MAX_TOKENS": ("16384", "max generate tokens (Qwen3 32B supports up to 32K)"),
    "COMPACTION_THRESHOLD": ("50", "conversation messages before compaction triggers"),
    "COMPACTION_KEEP_TAIL": ("16", "most recent messages kept verbatim after compaction"),
}

BACKUP_DIR = ROOT / "data" / "backups"


def _read_env() -> dict[str, str]:
    out: dict[str, str] = {}
    if not ENV_FILE.exists():
        return out
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


def _backup() -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dst = BACKUP_DIR / f".env-ollama-{ts}.bak"
    if ENV_FILE.exists():
        shutil.copy2(ENV_FILE, dst)
    return dst


def _write_env(env: dict[str, str]) -> None:
    lines: list[str] = []
    if ENV_FILE.exists():
        lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    keys = set(env)
    # Rebuild: keep original order, update values in place, append missing keys.
    out: list[str] = []
    seen: set[str] = set()
    header_done = False
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            k = stripped.partition("=")[0].strip()
            if k in env:
                if not header_done:
                    out.append("# --- switched by scripts/switch_ollama.py ---")
                    header_done = True
                out.append(f"{k}={env[k]}")
                seen.add(k)
                continue
        out.append(line)
    for k, v in env.items():
        if k not in seen:
            if not header_done:
                out.append("# --- switched by scripts/switch_ollama.py ---")
                header_done = True
            out.append(f"{k}={v}")
    ENV_FILE.write_text("\n".join(out) + "\n", encoding="utf-8")


def _mask(value: str) -> str:
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def _print_plan(env: dict[str, str], existing: dict[str, str]) -> None:
    print("Ollama switch preview (use --apply to write .env):")
    for k, (v, help) in OLLAMA_KEYS.items():
        old = existing.get(k, "<missing>")
        if "KEY" in k:
            old = _mask(old)
            v = _mask(v)
        mark = "" if old == v else "  << change"
        print(f"  {k:<20} {old:<28} -> {v}{mark}   # {help}")
    print()
    print("Reminder:")
    print("  1) Pull models first:  ollama pull qwen3:32b && ollama pull bge-m3")
    print("  2) If EMBEDDING_DIM changed or first switch: delete "
          "data/milvus_demo.db and re-ingest:")
    print("     python scripts/run_ingest.py --full-refresh")


def _restore() -> None:
    backups = sorted(BACKUP_DIR.glob(".env-ollama-*.bak")) if BACKUP_DIR.exists() else []
    if not backups:
        print("No backups found under data/backups/.")
        sys.exit(1)
    latest = backups[-1]
    shutil.copy2(latest, ENV_FILE)
    print(f"Restored .env from {latest}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Switch .env to intranet Ollama")
    ap.add_argument("--apply", action="store_true", help="write .env (default: preview only)")
    ap.add_argument("--restore", action="store_true", help="restore latest backup")
    ap.add_argument("--host", default="http://127.0.0.1:11434", help="Ollama base URL")
    ap.add_argument("--model", default="qwen3:32b", help="Ollama chat model tag")
    ap.add_argument("--embed-model", default="bge-m3", help="Ollama embedding model")
    ap.add_argument("--embed-dim", default="1024", help="embedding dimension")
    ap.add_argument("--milvus-mode", default="lite", choices=["lite", "server"],
                    help="Milvus mode after switch")
    args = ap.parse_args()

    if args.restore:
        _restore()
        return

    host = args.host.rstrip("/")
    env = {
        "LLM_PROVIDER": "ollama",
        "LLM_MODEL": args.model,
        "LLM_BASE_URL": host,
        "LLM_API_KEY": "ollama",
        "EMBEDDING_PROVIDER": "openai_compatible",
        "EMBEDDING_MODEL": args.embed_model,
        "EMBEDDING_BASE_URL": f"{host}/v1",
        "EMBEDDING_API_KEY": "ollama",
        "EMBEDDING_DIM": args.embed_dim,
        "MILVUS_MODE": args.milvus_mode,
    }
    existing = _read_env()
    _print_plan(env, existing)

    if args.apply:
        backup = _backup()
        _write_env(env)
        print(f"\n.env updated (backup: {backup})")
    else:
        print("\nDry-run only. Re-run with --apply to write .env.")


if __name__ == "__main__":
    main()
