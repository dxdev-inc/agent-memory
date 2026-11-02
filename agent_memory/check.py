"""Guard a memory directory.

Two failure modes make memories silently invisible:

1. Index truncation. Many agent tools load only the first 200 lines / 25 KB of
   `MEMORY.md` at session start; anything past that never reaches the model.
   This gives an explicit, scriptable check, with a tighter target so there is
   headroom.

2. Un-indexed topic files (orphans). A memory file that no index points at is
   found only by luck. The full catalog `INDEX.md` must list every topic file.

Two tiers:
  MEMORY.md  curated and always loaded; kept under the target so it fully loads.
  INDEX.md   the complete read-on-demand catalog of every memory, grouped by
             area. Not auto-loaded, so its size is unconstrained.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from .corpus import parse_frontmatter, topic_files

HARD_LINES = 200
HARD_BYTES = 25 * 1024
TARGET_LINES = 180
TARGET_BYTES = 17 * 1024

DEFAULT_AREA = "Other"
# Without a config file, the index is grouped by the `type` frontmatter value.
TYPE_AREAS = {
    "user": "About the user",
    "feedback": "Working rules and feedback",
    "project": "Projects",
    "reference": "References",
}


def load_areas(path: Path | None) -> list[tuple[str, list[str]]]:
    """Read an optional areas file: {"areas": {"Area name": ["keyword", ...]}}.

    Areas are tried in file order; the first area with a keyword found in the
    memory's name or filename wins. Returns [] when no file is given.
    """
    if path is None:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [(area, [k.lower() for k in kws])
            for area, kws in data.get("areas", {}).items()]


def classify(name: str, fname: str, mtype: str,
             areas: list[tuple[str, list[str]]]) -> str:
    key = f"{name} {fname}".lower()
    for area, kws in areas:
        if any(kw in key for kw in kws):
            return area
    if not areas:
        return TYPE_AREAS.get(mtype, DEFAULT_AREA)
    return DEFAULT_AREA


def _deslug(name: str) -> str:
    n = re.sub(r"^(feedback|reference|project|user)[-_]", "", name)
    return n.replace("_", " ").replace("-", " ").strip()


def rebuild_index(mem: Path, areas_file: Path | None = None) -> int:
    areas = load_areas(areas_file)
    entries = []
    for f in topic_files(mem):
        fm, _ = parse_frontmatter(f.read_text(encoding="utf-8"))
        name = (fm.get("name") or f.stem).strip()
        desc = (fm.get("description") or "").strip()
        if len(desc) > 200:
            desc = desc[:197].rstrip() + "..."
        entries.append((classify(name, f.name, fm.get("type", ""), areas),
                        f.name, _deslug(name), desc))

    by_area: dict[str, list] = {}
    for area, fname, title, desc in entries:
        by_area.setdefault(area, []).append((fname, title, desc))

    if areas:
        order = [a for a, _ in areas] + [DEFAULT_AREA]
    else:
        order = list(TYPE_AREAS.values()) + [DEFAULT_AREA]
    out = [
        "# Memory catalog (INDEX.md)\n",
        "Complete, read-on-demand catalog of every memory file, grouped by "
        "area. This file is not auto-loaded at session start (only "
        "`MEMORY.md` is, capped at 200 lines / 25 KB). Read it to check "
        "whether a memory about a topic already exists. The `description:` "
        "frontmatter of each file is the hook that makes a reader open it.\n",
        f"_{len(entries)} memories. Regenerate after adding or removing "
        f"files: `agent-memory rebuild-index`._\n",
    ]
    for area in order:
        items = by_area.get(area)
        if not items:
            continue
        out.append(f"\n## {area}  ({len(items)})\n")
        for fname, title, desc in sorted(items, key=lambda x: x[0]):
            line = f"- [{title}]({fname})"
            if desc:
                line += f": {desc}"
            out.append(line)
    text = "\n".join(out) + "\n"
    (mem / "INDEX.md").write_text(text, encoding="utf-8")
    print(f"Rebuilt INDEX.md: {len(entries)} entries, "
          f"{len(text.encode('utf-8'))} bytes")
    return 0


def past_limit(text: str, max_lines: int, max_bytes: int) -> list[tuple[int, str]]:
    """Lines of `text` that fall past `max_lines` or `max_bytes` (1-based).

    A line counts as past the byte limit when it does not fit whole inside the
    first `max_bytes` bytes. These are the entries that would be silently
    dropped at load time (hard limits) or have no headroom left (targets).
    """
    out: list[tuple[int, str]] = []
    used = 0
    for i, line in enumerate(text.split("\n"), start=1):
        used += len(line.encode("utf-8")) + 1
        if i > max_lines or used - 1 > max_bytes:
            out.append((i, line))
    return out


def _describe_past(rows: list[tuple[int, str]], limit: int = 12) -> list[str]:
    lines = [f"    line {n}: {(t[:100] + '...') if len(t) > 100 else t}"
             for n, t in rows if t.strip()][:limit]
    hidden = sum(1 for _, t in rows if t.strip()) - len(lines)
    if hidden > 0:
        lines.append(f"    ... and {hidden} more")
    return lines


def check(mem: Path) -> int:
    problems: list[str] = []
    memory_md = mem / "MEMORY.md"
    index_md = mem / "INDEX.md"

    if not memory_md.is_file():
        print(f"agent-memory: MEMORY.md not found at {memory_md}", file=sys.stderr)
        return 1

    mtext = memory_md.read_text(encoding="utf-8")
    mbytes = len(mtext.encode("utf-8"))
    mlines = mtext.count("\n") + 1

    def status(val, target, hard):
        if val > hard:
            return "OVER HARD LIMIT"
        if val > target:
            return "over target"
        return "ok"

    print(f"MEMORY.md: {mlines} lines (target {TARGET_LINES}, hard {HARD_LINES}) "
          f"[{status(mlines, TARGET_LINES, HARD_LINES)}]")
    print(f"MEMORY.md: {mbytes} bytes (target {TARGET_BYTES}, hard {HARD_BYTES}) "
          f"[{status(mbytes, TARGET_BYTES, HARD_BYTES)}]")
    if mlines > HARD_LINES or mbytes > HARD_BYTES:
        dropped = past_limit(mtext, HARD_LINES, HARD_BYTES)
        print("Entries NOT LOADED at session start (past the hard limit):")
        print("\n".join(_describe_past(dropped)))
        problems.append("MEMORY.md exceeds the load limit, so entries past it "
                        "silently drop. Compact it.")
    elif mlines > TARGET_LINES or mbytes > TARGET_BYTES:
        risk = past_limit(mtext, TARGET_LINES, TARGET_BYTES)
        print("Entries past the target (first to drop if MEMORY.md keeps growing):")
        print("\n".join(_describe_past(risk)))
        problems.append("MEMORY.md is over target (headroom gone). Compact soon.")

    files = topic_files(mem)
    print(f"Topic files: {len(files)}")

    # Every topic file must be referenced by INDEX.md.
    if not index_md.is_file():
        problems.append("INDEX.md missing. Run `agent-memory rebuild-index`.")
    else:
        itext = index_md.read_text(encoding="utf-8")
        refs = set(re.findall(r"\]\(([^)]+\.md)\)", itext))
        orphans = [f.name for f in files if f.name not in refs]
        dangling = [r for r in refs if not (mem / r).is_file()]
        print(f"INDEX.md references: {len(refs)} "
              f"(orphans {len(orphans)}, dangling {len(dangling)})")
        if orphans:
            problems.append(f"{len(orphans)} topic file(s) not in INDEX.md "
                            f"(invisible): {', '.join(sorted(orphans)[:10])}"
                            + (" ..." if len(orphans) > 10 else "")
                            + "  (run rebuild-index)")
        if dangling:
            problems.append(f"{len(dangling)} INDEX.md link(s) point at missing "
                            f"files: {', '.join(sorted(dangling)[:10])}")

    # Dangling [[wikilinks]] in bodies. A target is valid if it matches a topic
    # file's `name:` or its filename stem. Informational only: it never fails
    # the check, but it surfaces the rot a deletion leaves behind.
    valid_targets = {f.stem for f in files}
    texts = {f: f.read_text(encoding="utf-8") for f in files}
    for f, t in texts.items():
        fm, _ = parse_frontmatter(t)
        if fm.get("name"):
            valid_targets.add(fm["name"].strip())
    unresolved: dict[str, int] = {}
    for t in texts.values():
        for w in re.findall(r"\[\[([^\]]+)\]\]", t):
            w = w.strip()
            if w not in valid_targets:
                unresolved[w] = unresolved.get(w, 0) + 1
    if unresolved:
        names = ", ".join(sorted(unresolved)[:10])
        print(f"Body wikilinks: {len(unresolved)} unresolved [[..]] target(s) "
              f"(informational): {names}")

    # MEMORY.md links must resolve too.
    mrefs = set(re.findall(r"\]\(([^)]+\.md)\)", mtext))
    mdangling = [r for r in mrefs if not (mem / r).is_file()]
    if mdangling:
        problems.append(f"{len(mdangling)} MEMORY.md link(s) point at missing "
                        f"files: {', '.join(sorted(mdangling)[:10])}")

    # Every topic file needs name + description.
    missing_fm = []
    for f, t in texts.items():
        fm, _ = parse_frontmatter(t)
        if not fm.get("name") or not fm.get("description"):
            missing_fm.append(f.name)
    if missing_fm:
        problems.append(f"{len(missing_fm)} topic file(s) missing name/description "
                        f"frontmatter (weak recall): {', '.join(missing_fm[:10])}")

    print()
    if problems:
        print(f"FAIL: {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("OK: memory index healthy.")
    return 0
