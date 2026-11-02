"""Task-hint recall over a memory directory.

    agent-memory recall "<task hint>" --memory-dir DIR [--sid ID] [-n 5]

Local lexical TF-IDF over the curated fields (slug, name, description) with a
smaller weight on the body, plus a small boost for memories wikilinked from a
strong hit. No embeddings, no network, no model call. The index is rebuilt on
every call, so it can never go stale.

Properties: conservative (a few hits, never a page), task-driven (the hint),
and optionally session-aware (a seen-set per session id suppresses a memory
already surfaced in that session).
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .corpus import MemoryFile
from .similarity import (Vectors, build_index, cosine, slug_tokens, tokens,
                         wikilinks)

DEFAULT_N = 5
# A hit must clear this to be returned at all: "nothing relevant" is a valid
# answer, and a recall hook must not push noise into a session.
MIN_SCORE = 0.12
W_CURATED = 1.0
W_FULL = 0.35
LINK_BOOST = 0.15           # added to a memory wikilinked FROM a strong hit
LINK_FROM_SCORE = 0.35      # what counts as strong for the boost


@dataclass
class Hit:
    slug: str
    name: str
    description: str
    path: str
    score: float
    age_days: int

    def as_dict(self) -> dict:
        return {"slug": self.slug, "name": self.name,
                "description": self.description, "path": self.path,
                "score": round(self.score, 3), "age_days": self.age_days}


def _curated_vecs(memories: list[MemoryFile]) -> Vectors:
    """An index over slug + name + description only (body left out)."""
    stripped = [MemoryFile(path=m.path, slug=m.slug, name=m.name,
                           description=m.description, kind=m.kind,
                           body=m.name, mtime=m.mtime) for m in memories]
    return build_index(stripped)


def _query_vec(text: str, index: Vectors, memories_n: int) -> dict[str, float]:
    counts: dict[str, int] = {}
    for t in tokens(text) + sorted(slug_tokens(text)):
        counts[t] = counts.get(t, 0) + 1
    v: dict[str, float] = {}
    for term, tf in counts.items():
        df = index.df.get(term)
        if not df:
            continue
        v[term] = (1.0 + math.log(tf)) * (math.log(memories_n / df) + 1.0)
    norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {t: x / norm for t, x in v.items()}


def score_all(text: str, memories: list[MemoryFile]) -> dict[str, float]:
    """slug -> relevance of `text` to each memory."""
    if not memories:
        return {}
    cur, full = _curated_vecs(memories), build_index(memories)
    qc = _query_vec(text, cur, cur.n)
    qf = _query_vec(text, full, full.n)
    scores = {m.slug: W_CURATED * cosine(qc, cur.vecs[m.slug])
              + W_FULL * cosine(qf, full.vecs[m.slug]) for m in memories}
    by_slug = {m.slug: m for m in memories}
    boost: dict[str, float] = {}
    for slug, s in scores.items():
        if s >= LINK_FROM_SCORE:
            for tgt in wikilinks(by_slug[slug]):
                if tgt in scores and tgt != slug:
                    boost[tgt] = LINK_BOOST
    for tgt, b in boost.items():
        if scores[tgt] > 0:
            scores[tgt] += b
    return scores


def _hit(m: MemoryFile, score: float, now: float) -> Hit:
    return Hit(slug=m.slug, name=m.name, description=m.description,
               path=str(m.path), score=score,
               age_days=max(0, int((now - m.mtime) // 86400)))


# ---- seen-set -------------------------------------------------------------

def _seen_path(seen_dir: Path, sid: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_\-]", "_", sid)
    return Path(seen_dir) / f"{safe}.json"


def load_seen(seen_dir: Path, sid: str) -> set[str]:
    try:
        return set(json.loads(_seen_path(seen_dir, sid).read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return set()


def save_seen(seen_dir: Path, sid: str, seen: set[str]) -> None:
    p = _seen_path(seen_dir, sid)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sorted(seen)), encoding="utf-8")


# ---- recall ---------------------------------------------------------------

def recall(hint: str, memories: list[MemoryFile], *, n: int = DEFAULT_N,
           seen: set[str] | None = None, min_score: float = MIN_SCORE,
           now: float | None = None) -> list[Hit]:
    """Top-`n` memories for `hint`, skipping slugs in `seen`."""
    now = time.time() if now is None else now
    scores = score_all(hint, memories)
    by_slug = {m.slug: m for m in memories}
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    out: list[Hit] = []
    for slug, s in ranked:
        if s < min_score or len(out) >= n:
            break
        if seen and slug in seen:
            continue
        out.append(_hit(by_slug[slug], s, now))
    return out
