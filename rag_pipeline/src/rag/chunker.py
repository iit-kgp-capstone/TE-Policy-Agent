import re
from dataclasses import dataclass, field
from typing import Any

TABLE_BLOCK_RE = re.compile(r"((?:^\|.*\|$\n?)+)", re.MULTILINE)
HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$", re.MULTILINE)


@dataclass
class PolicyChunk:
    chunk_id: str
    text: str
    token_estimate: int
    has_table: bool
    metadata: dict[str, Any] = field(default_factory=dict)


def estimate_tokens(text: str) -> int:
    """Fast heuristic (~0.75 words/token) to keep chunks within [min, max] range."""
    return max(1, int(len(text.split()) * 1.33))


def split_preserving_tables(body: str) -> list[tuple[str, bool]]:
    """Splits text into (segment, is_table) pairs so tables are never broken."""
    segments: list[tuple[str, bool]] = []
    last_idx = 0
    for match in TABLE_BLOCK_RE.finditer(body):
        pre = body[last_idx : match.start()].strip()
        if pre:
            segments.append((pre, False))
        segments.append((match.group(1).strip(), True))
        last_idx = match.end()
    tail = body[last_idx:].strip()
    if tail:
        segments.append((tail, False))
    return segments


def semantic_split_prose(text: str, min_tokens: int = 80, max_tokens: int = 450) -> list[str]:
    """
    Groups paragraphs/sentences into variable-size semantic units within [min_tokens, max_tokens].
    Never slices mid-sentence.
    """
    if estimate_tokens(text) <= max_tokens:
        return [text]

    # Split by paragraph first, then sentence if a single paragraph is oversized
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for para in paragraphs:
        if estimate_tokens(para) <= max_tokens:
            units.append(para)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            units.extend([s.strip() for s in sentences if s.strip()])

    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    for unit in units:
        u_tokens = estimate_tokens(unit)
        if buf and (buf_tokens + u_tokens > max_tokens) and buf_tokens >= min_tokens:
            chunks.append(" ".join(buf))
            buf = [unit]
            buf_tokens = u_tokens
        else:
            buf.append(unit)
            buf_tokens += u_tokens

    if buf:
        chunks.append(" ".join(buf))
    return chunks


def chunk_policy_document(
    doc: dict[str, Any],
    min_tokens: int = 80,
    max_tokens: int = 450,
) -> list[PolicyChunk]:
    """
    Converts a corpus document (JSONL record or Markdown policy) into
    header-aware, table-preserving semantic chunks.
    """
    doc_id = doc["doc_id"]
    policy_ref = doc["policy_ref"]
    title = doc["title"]
    header_prefix = f"[{policy_ref}] {title}\n"

    raw_text = doc["text"]
    segments = split_preserving_tables(raw_text)

    chunks: list[PolicyChunk] = []
    chunk_idx = 0

    for seg_text, is_table in segments:
        if is_table:
            # Keep the entire rate table intact with its header
            full_text = f"{header_prefix}{seg_text}"
            chunks.append(
                PolicyChunk(
                    chunk_id=f"{doc_id}#c{chunk_idx:02d}",
                    text=full_text,
                    token_estimate=estimate_tokens(full_text),
                    has_table=True,
                )
            )
            chunk_idx += 1
        else:
            for prose_unit in semantic_split_prose(seg_text, min_tokens, max_tokens):
                full_text = f"{header_prefix}{prose_unit}"
                chunks.append(
                    PolicyChunk(
                        chunk_id=f"{doc_id}#c{chunk_idx:02d}",
                        text=full_text,
                        token_estimate=estimate_tokens(full_text),
                        has_table=False,
                    )
                )
                chunk_idx += 1

    return chunks