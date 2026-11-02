"""Candidate-pair generation for near-duplicate detection. Lexical, local, no
embeddings and no dependencies.

Why not embeddings: this layer only needs recall, not precision. It nominates
pairs for a person to read, so a generous threshold that nominates a few
unrelated pairs costs one extra line in a proposal. Memory folders are also
usually written in one voice about a closed set of topics, which is exactly
where IDF-weighted word overlap works well. A metered API or a local model
would add cost or infrastructure for recall the folder does not need.

The blind spot, stated plainly: two memories that say the same thing in
entirely different words are missed. `report_blind_spot()` exists so a proposal
says this instead of letting "no duplicates found" read as proof there are none.

Four channels feed the candidate set:
  tfidf  IDF-weighted cosine over name, description and body tokens.
  slug   shared words in the filename, which is a curated one-line summary.
  rare   two files sharing a term that appears in only a few files.
  link   one file's body links the other with [[slug]]. Not a similarity
         measure; an author's own statement that the two belong together.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from .corpus import KINDS, MemoryFile

_WORD_RE = re.compile(r"[a-z0-9_]+")

STOPWORDS = frozenset("""
a an the and or but if then than that this these those is are was were be been
being do does did done has have had having not no never always it its it's of
to in on at by for with from as so such use used using when where what which
who whom how why can could should would will shall may might must one two i me
my we us our you your he she they them his her their there here now then also
into over under out up down off about after before again more most less least
same other another each any all both few own too very just only even still yet
per via vs etc ie eg memory file files line lines
""".split())

# A term in more than this fraction of the folder is skipped when building the
# inverted index (still counted for IDF weighting).
MAX_DF_FRACTION = 0.08
# A term in at most this many files is "rare" and pairs its holders directly.
RARE_DF = 4
MIN_TOKEN_LEN = 3

# Field weights: slug and description are curated summaries, body is prose.
W_SLUG = 4
W_DESC = 2
W_BODY = 1


def tokens(text: str) -> list[str]:
    return [t for t in _WORD_RE.findall(text.lower())
            if len(t) >= MIN_TOKEN_LEN and t not in STOPWORDS and not t.isdigit()]


def slug_tokens(slug: str) -> set[str]:
    """The filename's own words. Underscores separate words here. The type
    prefix (feedback, project, ...) says nothing about the topic, so it is
    dropped."""
    parts = re.split(r"[^a-z0-9]+", slug.lower())
    return {p for p in parts
            if len(p) >= MIN_TOKEN_LEN and p not in STOPWORDS
            and p not in KINDS and not p.isdigit()}


def wikilinks(m: MemoryFile) -> set[str]:
    """Slugs this memory explicitly points at with `[[slug]]`."""
    return set(re.findall(r"\[\[([A-Za-z0-9_\-]+?)(?:\.md)?\]\]", m.body))


def _weighted_counts(m: MemoryFile) -> Counter:
    c: Counter = Counter()
    for t in slug_tokens(m.slug):
        c[t] += W_SLUG
    for t in tokens(m.description):
        c[t] += W_DESC
    for t in tokens(m.body):
        c[t] += W_BODY
    return c


@dataclass
class Vectors:
    """The whole lexical index. Small enough to rebuild on every run, so it can
    never go stale."""
    vecs: dict[str, dict[str, float]]      # slug -> term -> l2-normalised tf-idf
    df: dict[str, int]
    postings: dict[str, list[str]]         # term -> slugs, common terms dropped
    n: int


def build_index(memories: list[MemoryFile]) -> Vectors:
    counts = {m.slug: _weighted_counts(m) for m in memories}
    df: Counter = Counter()
    for c in counts.values():
        df.update(c.keys())
    n = max(len(memories), 1)

    vecs: dict[str, dict[str, float]] = {}
    for slug, c in counts.items():
        v: dict[str, float] = {}
        for term, tf in c.items():
            idf = math.log(n / df[term]) + 1.0
            v[term] = (1.0 + math.log(tf)) * idf
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs[slug] = {t: x / norm for t, x in v.items()}

    cap = max(int(n * MAX_DF_FRACTION), 2)
    postings: dict[str, list[str]] = defaultdict(list)
    for slug, c in counts.items():
        for term in c:
            if df[term] <= cap:
                postings[term].append(slug)
    return Vectors(vecs=vecs, df=dict(df), postings=dict(postings), n=n)


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(b) < len(a):
        a, b = b, a
    return sum(w * b[t] for t, w in a.items() if t in b)


@dataclass
class Candidate:
    a: str
    b: str
    score: float
    channels: list[str]
    shared_rare: list[str]


# Admission rule. A pair is nominated if any row matches. The numbers are
# loose on purpose: a real duplicate should rarely fall through, and a person
# reads every nomination anyway.
ADMIT_TFIDF = 0.30          # cosine alone is convincing
ADMIT_SLUG = 2              # at least N shared filename words
ADMIT_RARE_STRONG = 3       # at least N shared rare terms ...
ADMIT_RARE_FLOOR = 0.15     # ... and at least this much overall overlap
ADMIT_RARE_PAIR = 2         # 2 shared rare terms needs a higher floor
ADMIT_RARE_PAIR_FLOOR = 0.25
# `link` never admits alone: cross-references usually mean "related", not "the
# same". It corroborates, and admits only with real overlap behind it.
ADMIT_LINK_FLOOR = 0.22


def candidate_pairs(memories: list[MemoryFile], index: Vectors,
                    *, threshold: float = ADMIT_TFIDF,
                    slug_overlap: int = ADMIT_SLUG,
                    limit: int | None = None) -> list[Candidate]:
    """Pairs worth a person's attention, most similar first.

    Scanning is driven by the inverted index, so it is linear in co-occurring
    term pairs rather than quadratic in files. Nothing is cached between runs.
    """
    by_slug = {m.slug: m for m in memories}
    pair_rare: dict[tuple[str, str], set[str]] = defaultdict(set)
    seen: set[tuple[str, str]] = set()

    def _key(s1: str, s2: str) -> tuple[str, str]:
        return (s1, s2) if s1 < s2 else (s2, s1)

    # A term that is another memory's slug is a cross-reference, not a rare
    # coincidence; it feeds the `link` channel below instead.
    for term, slugs in index.postings.items():
        if len(slugs) < 2 or len(slugs) > 60:
            continue
        rare = len(slugs) <= RARE_DF and term not in by_slug
        for i, s1 in enumerate(slugs):
            for s2 in slugs[i + 1:]:
                key = _key(s1, s2)
                seen.add(key)
                if rare:
                    pair_rare[key].add(term)

    slug_post: dict[str, list[str]] = defaultdict(list)
    for m in memories:
        for t in slug_tokens(m.slug):
            slug_post[t].append(m.slug)
    slug_hits: Counter = Counter()
    for slugs in slug_post.values():
        if len(slugs) > 80:
            continue
        for i, s1 in enumerate(slugs):
            for s2 in slugs[i + 1:]:
                key = _key(s1, s2)
                slug_hits[key] += 1
                seen.add(key)

    linked: set[tuple[str, str]] = set()
    for m in memories:
        for target in wikilinks(m):
            if target in by_slug and target != m.slug:
                key = _key(m.slug, target)
                linked.add(key)
                seen.add(key)

    out: list[Candidate] = []
    for key in seen:
        s1, s2 = key
        score = cosine(index.vecs[s1], index.vecs[s2])
        rare = sorted(pair_rare.get(key, ()))
        channels: set[str] = set()
        if score >= threshold:
            channels.add("tfidf")
        if slug_hits.get(key, 0) >= slug_overlap:
            channels.add("slug")
        if (len(rare) >= ADMIT_RARE_STRONG and score >= ADMIT_RARE_FLOOR) or \
           (len(rare) >= ADMIT_RARE_PAIR and score >= ADMIT_RARE_PAIR_FLOOR):
            channels.add("rare")
        if key in linked:
            channels.add("link")
        if not channels or (channels == {"link"} and score < ADMIT_LINK_FLOOR):
            continue
        out.append(Candidate(a=s1, b=s2, score=round(score, 4),
                             channels=sorted(channels), shared_rare=rare[:8]))

    # Rank by cosine, with a bonus for each independent channel that agreed.
    out.sort(key=lambda c: (-(c.score + 0.05 * len(c.channels)), c.a, c.b))
    if limit is not None:
        out = out[:limit]
    return out


BLIND_SPOT = (
    "Duplicate candidates are found by IDF-weighted lexical overlap, not by "
    "meaning. Two memories that state the same rule with no words in common "
    "are NOT in this proposal and were never looked at. Read an empty "
    "duplicate section as 'the lexical pass found none', never as 'there are "
    "none'."
)


def report_blind_spot() -> str:
    return BLIND_SPOT
