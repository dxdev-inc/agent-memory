"""`agent-memory propose`: nominate near-duplicates, never destroy anything.

A read-only pass over a memory directory that writes one artifact, a proposal
folder holding `proposal.md` (for a person to edit) and `proposal.json` (for
`apply`). It does not create, edit, move or delete a memory. There is no model
call: candidates come from the lexical pass in `similarity.py`.

Each nominated pair becomes one `retire` item that proposes dropping the
smaller file and keeping the larger one. A retire loses whatever the dropped
file said that the kept one does not, so read both before marking it `apply`,
and fold anything unique into the kept file first. A model-assisted version
could draft merged bodies; this one does not.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import similarity
from .apply import (SNAPSHOT, ApplyError, check_run_id,
                    check_runs_outside_memory)
from .corpus import MemoryFile, load_dir

APPLICABLE = {"merge", "retire"}
DEFAULT_RUNS_DIR = ".agent-memory-runs"


class ProposeError(RuntimeError):
    pass


@dataclass
class Item:
    id: str
    rot_class: str
    action: str                       # merge | retire
    files: list[str]
    reason: str
    evidence: dict = field(default_factory=dict)
    keep: str | None = None

    def as_dict(self) -> dict:
        return {"id": self.id, "class": self.rot_class, "action": self.action,
                "files": self.files, "reason": self.reason,
                "evidence": self.evidence, "keep": self.keep,
                "applicable": self.action in APPLICABLE}


def _which_to_keep(a: MemoryFile, b: MemoryFile) -> tuple[MemoryFile, MemoryFile]:
    """(keep, drop): the larger body wins, then the newer file, then slug."""
    ranked = sorted([a, b], key=lambda m: (-m.bytes, -m.mtime, m.slug))
    return ranked[0], ranked[1]


def propose(memory_dir: Path, *, runs_dir: Path | None = None,
            run_id: str | None = None, max_pairs: int = 50,
            log=print) -> tuple[Path, dict]:
    """Analyse `memory_dir` and write a proposal. Returns (run_dir, summary)."""
    shown_dir = Path(memory_dir)
    memory_dir = shown_dir.resolve()
    memories = load_dir(memory_dir)
    by_slug = {m.slug: m for m in memories}
    log(f"memory: {len(memories)} files in {shown_dir.as_posix()}")

    index = similarity.build_index(memories)
    cands = similarity.candidate_pairs(memories, index)
    log(f"memory: {len(cands)} near-duplicate candidate pair(s) (lexical)")

    items: list[Item] = []
    used: set[str] = set()
    for c in cands[:max_pairs]:
        # One decision per file: a file already in an earlier item waits for
        # the next run, after that item has been reviewed.
        if c.a in used or c.b in used:
            continue
        keep, drop = _which_to_keep(by_slug[c.a], by_slug[c.b])
        used.update((c.a, c.b))
        items.append(Item(
            id=f"M-{len(items) + 1:03d}",
            rot_class="duplicate",
            action="retire",
            files=[drop.slug],
            reason=(f"{drop.slug} overlaps {keep.slug}; keeping the larger "
                    f"file. Read both before applying."),
            evidence={"cosine": c.score, "channels": c.channels,
                      "shared_rare_terms": c.shared_rare,
                      "bytes": {c.a: by_slug[c.a].bytes, c.b: by_slug[c.b].bytes}},
            keep=keep.slug))

    stamp = run_id if run_id is not None else datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = (Path(runs_dir) if runs_dir else Path(DEFAULT_RUNS_DIR)) / stamp
    try:
        check_run_id(stamp)
        check_runs_outside_memory(run_dir, memory_dir)
    except ApplyError as e:
        raise ProposeError(str(e)) from None
    # A run that was already applied owns a snapshot (and retired files) that
    # `apply --undo` restores from. Writing a new proposal over it would leave
    # that snapshot stale, so an applied run id is never reused.
    if (run_dir / SNAPSHOT).exists() or (memory_dir / ".retired" / stamp).exists():
        raise ProposeError(
            f"run id '{stamp}' was already applied and its undo snapshot is kept. "
            f"Pick a new --run-id, or remove {run_dir} and "
            f"{memory_dir / '.retired' / stamp} yourself if you no longer need to undo it.")
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "memory_dir": str(memory_dir),
        "generated": datetime.now().isoformat(timespec="seconds"),
        "files": len(memories),
        "candidate_pairs": len(cands),
        "items": len(items),
        "applicable": sum(1 for i in items if i.action in APPLICABLE),
    }
    payload = {"summary": summary, "items": [i.as_dict() for i in items]}
    (run_dir / "proposal.json").write_text(json.dumps(payload, indent=2),
                                           encoding="utf-8")
    (run_dir / "proposal.md").write_text(
        render_proposal(summary, items, memories, run_dir), encoding="utf-8")
    log(f"memory: proposal written to {(run_dir / 'proposal.md').as_posix()}")
    return run_dir, summary


HEADER = """\
# Memory consolidation proposal

**Nothing here has happened.** This file is a proposal over
`{memory_dir}`. No memory was created, edited, moved or deleted to produce it.

{files} memories read, {items} proposal(s), {applicable} applicable.

## How to review

Each item below carries a `decision:` line. Change it and save this file:

* `decision: hold`   the default. Apply skips it.
* `decision: apply`  do it. Only `merge` and `retire` items can be applied.
* `decision: reject` wrong call; apply skips it.

Then run:

```
agent-memory apply {run_dir}           # dry run: prints what it would do
agent-memory apply {run_dir} --yes     # execute
agent-memory apply {run_dir} --undo --yes   # put everything back
```

Apply never deletes. A retired memory moves to `.retired/<run-id>/` inside the
memory directory, with a manifest, and `--undo` restores the whole directory
from a snapshot taken before the first change.

### What this pass cannot see

{blind_spot}
"""


def render_proposal(summary: dict, items: list[Item],
                    memories: list[MemoryFile], run_dir: Path) -> str:
    by_slug = {m.slug: m for m in memories}
    parts = [HEADER.format(**summary, run_dir=Path(run_dir).as_posix(),
                           blind_spot=similarity.report_blind_spot())]
    if not items:
        parts.append("\n---\n\nNo proposals. The lexical pass found no "
                     "near-duplicates.\n")
    for i in items:
        parts.append(_render_item(i, by_slug))
    return "\n".join(parts) + "\n"


def _render_item(i: Item, by_slug: dict[str, MemoryFile]) -> str:
    out = ["\n---\n", f"### {i.id}  `{i.action}`  {' + '.join(i.files)}", ""]
    out.append(f"**Why:** {i.reason}")
    ev = i.evidence
    if "cosine" in ev:
        line = (f"**Evidence:** cosine {ev['cosine']}, channels "
                f"{', '.join(ev.get('channels', []))}")
        if ev.get("shared_rare_terms"):
            line += f", shared rare terms: {', '.join(ev['shared_rare_terms'])}"
        out.append(line)
    if i.keep:
        out.append(f"**Keeps:** `{i.keep}`")
    for slug in ([i.keep] if i.keep else []) + i.files:
        m = by_slug.get(slug)
        if m:
            out.append(f"  - `{slug}` ({m.bytes} B): {m.description[:220]}")
    out.append("")
    out.append(f"decision: hold    <!-- {i.id}: hold | apply | reject -->")
    return "\n".join(out)
