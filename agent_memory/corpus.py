"""Load a memory directory into comparable records.

A memory directory is a flat folder of `<slug>.md` files, each with `name` and
`description` frontmatter, plus two index files: `MEMORY.md` (the short,
always-loaded tier) and `INDEX.md` (the full catalog). The index files are not
memories and are excluded from every pass here.

This module only reads and parses. No judgment lives here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

INDEX_FILES = {"MEMORY.md", "INDEX.md", "README.md"}
KINDS = {"feedback", "reference", "project", "user"}

_FM_RE = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n?", re.DOTALL)
_COMMENT_RE = re.compile(r"\s+#.*$")
_DATE_RE = re.compile(r"\b(20\d\d)-(\d\d)-(\d\d)\b")


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Return (flat frontmatter dict, body). Nested keys are flattened to the
    leaf name, so `type` under `metadata:` is just `type`."""
    m = _FM_RE.match(text)
    if not m:
        return {}, text
    fm: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k = k.strip().lstrip("-").strip()
        v = v.strip()
        if v[:1] not in ('"', "'"):
            v = _COMMENT_RE.sub("", v)  # YAML: ` #` starts a comment on an unquoted value
        v = v.strip().strip('"').strip("'")
        if k and v and k not in fm:
            fm[k] = v
    return fm, text[m.end():]


def _dates(text: str) -> list[str]:
    out = []
    for m in _DATE_RE.finditer(text):
        try:
            out.append(date(int(m.group(1)), int(m.group(2)),
                            int(m.group(3))).isoformat())
        except ValueError:
            continue
    return sorted(set(out))


@dataclass
class MemoryFile:
    path: Path
    slug: str                      # filename without .md, the stable id
    name: str
    description: str
    kind: str                      # feedback | reference | project | user | other
    body: str
    mtime: float
    dates: list[str] = field(default_factory=list)
    in_core: bool = False          # linked from MEMORY.md

    @property
    def bytes(self) -> int:
        return len(self.body.encode("utf-8"))


def _kind(slug: str, declared: str) -> str:
    if declared in KINDS:
        return declared
    head = slug.split("_", 1)[0].split("-", 1)[0]
    return head if head in KINDS else "other"


def topic_files(memory_dir: Path) -> list[Path]:
    return sorted(p for p in Path(memory_dir).glob("*.md")
                  if p.name not in INDEX_FILES)


def load_dir(memory_dir: Path) -> list[MemoryFile]:
    """Every memory in `memory_dir`, sorted by slug. Index files excluded."""
    memory_dir = Path(memory_dir)
    if not memory_dir.is_dir():
        raise FileNotFoundError(f"not a memory directory: {memory_dir}")

    core_text = ""
    core_path = memory_dir / "MEMORY.md"
    if core_path.exists():
        core_text = core_path.read_text(encoding="utf-8", errors="replace")

    out: list[MemoryFile] = []
    for p in topic_files(memory_dir):
        text = p.read_text(encoding="utf-8", errors="replace")
        fm, body = parse_frontmatter(text)
        out.append(MemoryFile(
            path=p,
            slug=p.stem,
            name=fm.get("name", p.stem),
            description=fm.get("description", ""),
            kind=_kind(p.stem, fm.get("type", "")),
            body=body.strip(),
            mtime=p.stat().st_mtime,
            dates=_dates(body),
            in_core=f"({p.name})" in core_text,
        ))
    return out
