"""Split plain text into overlapping chunks for embedding."""
from __future__ import annotations

import re


def extract_text_from_bytes(data: bytes, *, filename: str, kind: str) -> str:
    """Decode upload bytes to text. PDF uses pypdf when available."""
    name = (filename or "").lower()
    k = (kind or "").lower()
    if k == "pdf" or name.endswith(".pdf"):
        return _extract_pdf(data)
    # utf-8 with BOM / gb fallback
    for enc in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _extract_pdf(data: bytes) -> str:
    try:
        from io import BytesIO

        from pypdf import PdfReader

        reader = PdfReader(BytesIO(data))
        parts: list[str] = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                continue
        text = "\n".join(parts).strip()
        if not text:
            raise ValueError("PDF 未能提取到文本（可能是扫描件）")
        return text
    except ImportError as e:
        raise ValueError("服务端未安装 pypdf，无法解析 PDF；请 pip install pypdf") from e


def chunk_text(text: str, *, size: int = 900, overlap: int = 120) -> list[str]:
    """Paragraph-aware chunking with character overlap."""
    raw = re.sub(r"\r\n?", "\n", text or "").strip()
    if not raw:
        return []
    # Prefer split on blank lines
    paras = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    if not paras:
        paras = [raw]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if not buf:
            buf = p
        elif len(buf) + 2 + len(p) <= size:
            buf = buf + "\n\n" + p
        else:
            chunks.extend(_flush_long(buf, size=size, overlap=overlap))
            buf = p
    if buf:
        chunks.extend(_flush_long(buf, size=size, overlap=overlap))
    return [c for c in chunks if c.strip()]


def _flush_long(text: str, *, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text]
    out: list[str] = []
    step = max(1, size - overlap)
    i = 0
    while i < len(text):
        out.append(text[i : i + size].strip())
        if i + size >= len(text):
            break
        i += step
    return [c for c in out if c]
