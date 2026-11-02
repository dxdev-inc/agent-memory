"""`agent-memory apply`: execute a reviewed proposal, reversibly.

Five properties, in the order they matter. A memory is accumulated judgment;
silently losing one is worse than keeping a duplicate forever.

1. Apply never deletes. A retired memory moves to
   `<memory-dir>/.retired/<run-id>/` next to a `MANIFEST.json` saying which
   item retired it and why. Only `undo_run` unlinks a `.md`, and only after the
   current state has been copied aside, so undo is itself undoable.
2. Nothing happens without a human mark. Every item defaults to
   `decision: hold`. Apply reads the decisions out of `proposal.md`, the file a
   person edits, not the JSON. Zero `apply` marks means apply refuses.
3. Only two actions execute: `merge` and `retire`.
4. A full snapshot of the memory directory precedes the first change, and
   `--undo` restores it wholesale.
5. Dry run is the default. `--yes` is required to touch disk.

A `merge` item needs `merged_slug` and `merged_body` in `proposal.json`
(optionally `merged_description`). `propose` does not write them; they exist so
a hand-edited or tool-assisted proposal can carry a merged file.
"""
from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path

DECISIONS = {"hold", "apply", "reject"}
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]+(\.[A-Za-z0-9_-]+)*$")
SNAPSHOT = "snapshot-before"
_DECISION_RE = re.compile(r"^decision:\s*(\w+)\s*<!--\s*(M-\d+):", re.MULTILINE)


class ApplyError(RuntimeError):
    pass


def read_decisions(proposal_md: Path) -> dict[str, str]:
    """item id -> decision, parsed out of the file a human edits."""
    text = Path(proposal_md).read_text(encoding="utf-8", errors="replace")
    out: dict[str, str] = {}
    for value, item_id in _DECISION_RE.findall(text):
        v = value.strip().lower()
        out[item_id] = v if v in DECISIONS else "hold"
    return out


def load_run(run_dir: Path) -> tuple[dict, dict[str, str]]:
    run_dir = Path(run_dir)
    pj, pm = run_dir / "proposal.json", run_dir / "proposal.md"
    if not pj.exists() or not pm.exists():
        raise ApplyError(f"not a proposal run directory: {run_dir}")
    return json.loads(pj.read_text(encoding="utf-8")), read_decisions(pm)


def _frontmatter_for(slug: str, description: str, sources: list[str]) -> str:
    desc = description.replace('"', "'").strip()
    return ("---\n"
            f"name: {slug}\n"
            f'description: "{desc}"\n'
            "metadata:\n"
            f"  merged_from: {', '.join(sources)}\n"
            f"  merged_on: {datetime.now().date().isoformat()}\n"
            "---\n\n")


def check_slug(memory_dir: Path, slug: str) -> None:
    """Refuse a slug that could name a file outside `memory_dir`.

    A slug is a bare file name without its `.md`: no separators, no drive
    letter, no leading dot (so no `..`, no `.retired`)."""
    bad = (not isinstance(slug, str) or not slug or slug.startswith(".")
           or any(c in slug for c in '/\\:\0'))
    if not bad:
        target = (memory_dir / f"{slug}.md").resolve()
        bad = target.parent != memory_dir.resolve()
    if bad:
        raise ApplyError(f"refusing unsafe slug {slug!r}: a slug is a file name "
                         f"inside {memory_dir}, never a path. Nothing was touched.")


def check_run_id(run_id: str) -> None:
    """A run id is one folder name: letters, digits, `-`, `_`, and dots only
    between other characters. No separators, no `..`, not empty."""
    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        raise ApplyError(
            f"refusing run id {run_id!r}: use letters, digits, '-' and '_' "
            f"(dots only inside the name); no path separators, no '..', not empty.")


def check_runs_outside_memory(run_dir: Path, memory_dir: Path) -> None:
    """The run folder (and its snapshot) must not live inside the memory folder,
    or the snapshot would copy itself into itself."""
    run, mem = Path(run_dir).resolve(), Path(memory_dir).resolve()
    if run == mem or mem in run.parents:
        raise ApplyError(
            f"refusing: the runs folder {run} is inside the memory folder {mem}. "
            f"Use --runs-dir with a location outside the memory folder "
            f"(the default .agent-memory-runs is relative to the current "
            f"directory, so run from outside the memory folder). Nothing was touched.")


def plan(run_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """(to_execute, skipped_by_decision, not_applicable)."""
    payload, decisions = load_run(run_dir)
    execute, skipped, blocked = [], [], []
    for item in payload["items"]:
        decision = decisions.get(item["id"], "hold")
        row = item | {"decision": decision}
        if decision != "apply":
            skipped.append(row)
        elif not (item.get("applicable") or item.get("applyable")):
            blocked.append(row)
        else:
            execute.append(row)
    return execute, skipped, blocked


def apply_run(run_dir: Path, *, yes: bool = False, log=print) -> dict:
    run_dir = Path(run_dir)
    payload, _ = load_run(run_dir)
    memory_dir = Path(payload["summary"]["memory_dir"])
    if not memory_dir.is_dir():
        raise ApplyError(f"memory directory is gone: {memory_dir}")

    check_run_id(run_dir.resolve().name)
    check_runs_outside_memory(run_dir, memory_dir)
    execute, skipped, blocked = plan(run_dir)
    for row in execute:
        for slug in row["files"]:
            check_slug(memory_dir, slug)
        if row["action"] == "merge" and row.get("merged_slug"):
            check_slug(memory_dir, row["merged_slug"])
    for row in blocked:
        log(f"REFUSED {row['id']}: marked apply but action is "
            f"'{row['action']}', which this tool does not execute.")
    held = sum(1 for r in skipped if r["decision"] == "hold")
    rejected = sum(1 for r in skipped if r["decision"] == "reject")
    log(f"apply: {len(execute)} to execute, {held} held, {rejected} rejected, "
        f"{len(blocked)} refused")

    if not execute:
        log("apply: nothing marked `decision: apply`. Edit proposal.md first. "
            "Nothing was touched.")
        return {"executed": 0, "held": held, "rejected": rejected,
                "refused": len(blocked), "dry_run": not yes}

    run_id = run_dir.resolve().name
    if (run_dir / SNAPSHOT).exists() or (memory_dir / ".retired" / run_id / "MANIFEST.json").exists():
        raise ApplyError(
            f"run '{run_id}' was already applied; its manifest and undo snapshot "
            f"are kept. Run `agent-memory apply {run_dir} --undo --yes` to put it "
            f"back, then propose again with a new --run-id. Nothing was touched.")

    # Same refusals the real run would hit, so a dry run predicts them.
    refusals = {row["id"]: why for row in execute
                if (why := _merge_refusal(memory_dir, row))}
    ok_rows = [r for r in execute if r["id"] not in refusals]

    if not yes:
        for row in execute:
            if row["id"] in refusals:
                log(f"WOULD REFUSE {row['id']}: {refusals[row['id']]}")
                continue
            log(f"WOULD {row['action']:<7} {', '.join(row['files'])}"
                + (f"  -> {row.get('merged_slug')}.md"
                   if row["action"] == "merge" else ""))
        log("apply: dry run. Re-run with --yes to execute.")
        return {"executed": 0, "would_execute": len(ok_rows), "held": held,
                "rejected": rejected, "refused": len(blocked) + len(refusals),
                "dry_run": True}

    snapshot = run_dir / SNAPSHOT
    if not snapshot.exists():
        shutil.copytree(memory_dir, snapshot,
                        ignore=shutil.ignore_patterns(".retired"))
        log(f"apply: snapshot of {memory_dir} -> {snapshot}")

    retired_dir = memory_dir / ".retired" / run_id
    retired_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    done = 0
    refused = len(blocked)

    for row in execute:
        if row["id"] in refusals:
            log(f"REFUSED {row['id']}: {refusals[row['id']]}")
            refused += 1
            continue
        if row["action"] == "merge":
            target = memory_dir / f"{row['merged_slug']}.md"
            body = (_frontmatter_for(row["merged_slug"],
                                     row.get("merged_description", ""),
                                     row["files"])
                    + row["merged_body"].strip() + "\n")
            for slug in row["files"]:
                _retire(memory_dir / f"{slug}.md", retired_dir, row, manifest,
                        note=f"merged into {row['merged_slug']}.md", log=log)
            target.write_text(body, encoding="utf-8")
            log(f"MERGED  {', '.join(row['files'])} -> {target.name}")
            done += 1
        elif row["action"] == "retire":
            for slug in row["files"]:
                _retire(memory_dir / f"{slug}.md", retired_dir, row, manifest,
                        note=row["reason"], log=log)
            done += 1

    (retired_dir / "MANIFEST.json").write_text(
        json.dumps({"run": run_id, "memory_dir": str(memory_dir),
                    "applied": datetime.now().isoformat(timespec="seconds"),
                    "entries": manifest}, indent=2), encoding="utf-8")
    log(f"apply: {done} item(s) executed. Retired files in {retired_dir}")
    log(f"apply: undo with `agent-memory apply {run_dir} --undo --yes`")
    return {"executed": done, "held": held, "rejected": rejected,
            "refused": refused, "retired": len(manifest),
            "dry_run": False, "retired_dir": str(retired_dir)}


def _merge_refusal(memory_dir: Path, row: dict) -> str | None:
    """Why a merge item cannot run, or None. Shared by dry run and real run."""
    if row["action"] != "merge":
        return None
    if not row.get("merged_slug") or not row.get("merged_body"):
        return "merge item has no merged_slug / merged_body in proposal.json"
    target = memory_dir / f"{row['merged_slug']}.md"
    if target.exists() and row["merged_slug"] not in row["files"]:
        return (f"{target.name} already exists and is not one of the files "
                f"being merged; not overwriting it.")
    return None


def _retire(path: Path, retired_dir: Path, row: dict, manifest: list[dict],
            *, note: str, log=print) -> None:
    if not path.exists():
        log(f"SKIP    {path.name} is already gone; nothing to retire")
        return
    dest = retired_dir / path.name
    shutil.move(str(path), str(dest))
    manifest.append({"item": row["id"], "class": row["class"],
                     "file": path.name, "note": note,
                     "retired_to": str(dest)})
    log(f"RETIRED {path.name} -> {dest.parent.name}/")


def undo_run(run_dir: Path, *, yes: bool = False, log=print) -> dict:
    """Restore the memory directory from the snapshot this run took."""
    run_dir = Path(run_dir)
    payload, _ = load_run(run_dir)
    memory_dir = Path(payload["summary"]["memory_dir"])
    snapshot = run_dir / SNAPSHOT
    if not snapshot.is_dir():
        raise ApplyError(f"no snapshot in {run_dir}; this run never executed")
    files = sorted(snapshot.glob("*.md"))
    if not yes:
        log(f"undo: would restore {len(files)} files into {memory_dir} "
            f"from {snapshot}")
        log("undo: dry run. Re-run with --yes.")
        return {"restored": 0, "dry_run": True, "would_restore": len(files)}

    # Park whatever is there now, so undo is itself undoable.
    parked = run_dir / f"pre-undo-{datetime.now().strftime('%H%M%S')}"
    shutil.copytree(memory_dir, parked, ignore=shutil.ignore_patterns(".retired"))
    for p in memory_dir.glob("*.md"):
        p.unlink()
    for p in files:
        shutil.copy2(p, memory_dir / p.name)
    log(f"undo: restored {len(files)} files into {memory_dir}")
    log(f"undo: the state before this undo is parked at {parked}")
    return {"restored": len(files), "dry_run": False, "parked": str(parked)}
