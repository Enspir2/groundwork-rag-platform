"""
Chunking: splits raw document text into overlapping chunks suitable for embedding.

Why overlap matters: if a sentence explaining "why Europe declined" gets split exactly
at the chunk boundary, neither chunk alone captures the full context. A small overlap
(e.g. 50-100 chars) reduces the chance that meaning gets cut in half.

This is intentionally simple (paragraph + fixed-size fallback) rather than using a
library, so you can explain exactly how it works in an interview.
"""
from dataclasses import dataclass, field
from typing import List
import re
import os
import hashlib


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def _split_into_paragraphs(text: str) -> List[str]:
    # Split on blank lines (markdown paragraph boundaries), drop empties
    paras = [p.strip() for p in re.split(r"\n\s*\n", text)]
    return [p for p in paras if p]


def chunk_text(
    text: str,
    doc_id: str,
    doc_title: str,
    max_chunk_chars: int = 800,
    overlap_chars: int = 100,
) -> List[Chunk]:
    """
    Strategy:
    1. Split by paragraph first (respects natural document structure).
    2. If a paragraph is longer than max_chunk_chars, sub-split it with overlap.
    3. Small consecutive paragraphs get merged up to max_chunk_chars to avoid
       tiny, low-signal chunks.
    """
    paragraphs = _split_into_paragraphs(text)
    raw_chunks: List[str] = []
    buffer = ""

    for para in paragraphs:
        # Headings (markdown '#') get their own chunk boundary - keeps sections coherent
        if para.startswith("#"):
            if buffer:
                raw_chunks.append(buffer.strip())
                buffer = ""
            buffer = para
            continue

        candidate = (buffer + "\n\n" + para).strip() if buffer else para

        if len(candidate) <= max_chunk_chars:
            buffer = candidate
        else:
            if buffer:
                raw_chunks.append(buffer.strip())
            if len(para) > max_chunk_chars:
                # Sub-split long paragraph with overlap
                start = 0
                while start < len(para):
                    end = start + max_chunk_chars
                    raw_chunks.append(para[start:end].strip())
                    start = end - overlap_chars
                buffer = ""
            else:
                buffer = para

    if buffer:
        raw_chunks.append(buffer.strip())

    chunks: List[Chunk] = []
    for i, c in enumerate(raw_chunks):
        if not c:
            continue
        chunk_id = hashlib.md5(f"{doc_id}-{i}-{c[:30]}".encode()).hexdigest()[:12]
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                doc_title=doc_title,
                text=c,
                chunk_index=i,
                metadata={"source_file": doc_id},
            )
        )
    return chunks


def chunk_directory(dir_path: str) -> List[Chunk]:
    """Load every .md file in a directory and chunk it, tagging each chunk with
    which roles are allowed to see it, based on its source document."""
    # Permission mapping: which roles can see which documents. "all" means
    # every authenticated user, regardless of role, can see it.
    # This mapping is the single source of truth for document sensitivity —
    # centralizing it here (rather than scattering role checks elsewhere) is
    # what makes it hard to accidentally forget to protect a new document.
    DOC_PERMISSIONS = {
        "finance_q1_report": ["finance", "manager"],
        "finance_q2_report": ["finance", "manager"],
        "hr_policy_leave": ["hr", "manager"],
        "product_electronics_recall": ["all"],
    }

    all_chunks: List[Chunk] = []
    for fname in sorted(os.listdir(dir_path)):
        if not fname.endswith(".md"):
            continue
        fpath = os.path.join(dir_path, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read()
        title_line = text.strip().split("\n")[0].lstrip("# ").strip()
        doc_id = fname.replace(".md", "")
        allowed_roles = DOC_PERMISSIONS.get(doc_id, ["all"])  # default: visible to all if untagged

        chunks = chunk_text(text, doc_id=doc_id, doc_title=title_line)
        for c in chunks:
            c.metadata["allowed_roles"] = allowed_roles
        all_chunks.extend(chunks)
    return all_chunks


def filter_chunks_by_role(chunks: List[Chunk], user_role: str) -> List[Chunk]:
    """
    Returns only chunks the given role is authorized to see. This MUST be called
    before any ranking/retrieval happens — never rank first and filter after
    (see Phase 8 notes: filtering after ranking means restricted content gets
    scored/processed on behalf of an unauthorized request, which is the actual
    security violation, independent of whether the content is later hidden).
    """
    return [
        c for c in chunks
        if "all" in c.metadata.get("allowed_roles", ["all"]) or user_role in c.metadata.get("allowed_roles", [])
    ]


if __name__ == "__main__":
    # Quick manual test
    here = os.path.dirname(__file__)
    docs_dir = os.path.join(here, "..", "data", "docs")
    chunks = chunk_directory(docs_dir)
    print(f"Total chunks: {len(chunks)}\n")
    for c in chunks:
        print(f"[{c.doc_title}] chunk {c.chunk_index} ({len(c.text)} chars) id={c.chunk_id}")
        print("  ", c.text[:100].replace("\n", " "), "...")
        print()
