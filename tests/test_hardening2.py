"""Regression tests from the second pre-publication review: Windows device
names, bad merge filenames, a fully refused apply, undo scope, missing sources,
session ids that sanitize alike."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from agent_memory import apply as apply_mod
from agent_memory import propose as propose_mod
from agent_memory import recall as recall_mod
from agent_memory.cli import main

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "memory"


def quiet(*_):
    return None


@pytest.fixture
def mem(tmp_path: Path) -> Path:
    dest = tmp_path / "memory"
    shutil.copytree(EXAMPLE, dest)
    return dest


def _propose(mem, tmp_path, run_id="t"):
    run_dir, _ = propose_mod.propose(mem, runs_dir=tmp_path / "runs",
                                     run_id=run_id, log=quiet)
    return run_dir


def _mark(run_dir: Path) -> None:
    md = run_dir / "proposal.md"
    md.write_text(md.read_text().replace("decision: hold", "decision: apply"),
                  encoding="utf-8")


def _edit_item(run_dir: Path, **fields) -> None:
    pj = run_dir / "proposal.json"
    payload = json.loads(pj.read_text())
    payload["items"][0].update(fields)
    pj.write_text(json.dumps(payload), encoding="utf-8")


def _merge_into_existing(mem, tmp_path):
    run_dir = _propose(mem, tmp_path)
    payload = json.loads((run_dir / "proposal.json").read_text())
    other = next(p.stem for p in mem.glob("*.md")
                 if p.stem not in payload["items"][0]["files"]
                 and p.name not in ("MEMORY.md", "INDEX.md"))
    _edit_item(run_dir, action="merge", merged_slug=other,
               merged_body="merged body", applicable=True)
    _mark(run_dir)
    return run_dir


# ---- item 2: Windows device names ------------------------------------------

@pytest.mark.parametrize("bad", ["CON", "con", "NUL", "aux", "COM1", "lpt9",
                                 "CON.txt", "com3.v2"])
def test_reserved_device_names_are_refused_as_run_ids(mem, tmp_path, bad):
    with pytest.raises(propose_mod.ProposeError, match="device names"):
        propose_mod.propose(mem, runs_dir=tmp_path / "runs", run_id=bad, log=quiet)


@pytest.mark.parametrize("bad", ["CON", "nul", "COM1", "aux.v2"])
def test_reserved_device_names_are_refused_as_slugs(mem, bad):
    with pytest.raises(apply_mod.ApplyError):
        apply_mod.check_slug(mem, bad)


def test_cli_propose_filesystem_error_is_a_clean_refusal(mem, tmp_path, capsys):
    blocker = tmp_path / "runs"
    blocker.write_text("a file, not a folder", encoding="utf-8")
    rc = main(["propose", "--memory-dir", str(mem), "--runs-dir", str(blocker),
               "--run-id", "ok"])
    assert rc == 1
    assert "agent-memory propose:" in capsys.readouterr().err


# ---- item 3: bad merge filename refused before anything moves ---------------

@pytest.mark.parametrize("bad", ["bad*name", "a?b", 'q"q', "a<b", "a>b", "a|b",
                                 "a:b", "tab\tname", "trail."])
def test_bad_merged_slug_is_refused_before_anything_moves(mem, tmp_path, bad):
    run_dir = _propose(mem, tmp_path)
    _edit_item(run_dir, action="merge", merged_slug=bad, merged_body="body",
               applicable=True)
    _mark(run_dir)
    before = {p.name: p.read_bytes() for p in mem.glob("*.md")}
    with pytest.raises(apply_mod.ApplyError):
        apply_mod.apply_run(run_dir, yes=True, log=quiet)
    assert {p.name: p.read_bytes() for p in mem.glob("*.md")} == before
    assert not (mem / ".retired").exists()
    assert not (run_dir / "snapshot-before").exists()


def test_cli_apply_oserror_is_a_clean_error(mem, tmp_path, capsys, monkeypatch):
    run_dir = _propose(mem, tmp_path)
    _mark(run_dir)

    def boom(*a, **k):
        raise OSError("disk says no")
    monkeypatch.setattr(apply_mod, "apply_run", boom)
    assert main(["apply", str(run_dir), "--yes"]) == 1
    assert "disk says no" in capsys.readouterr().err


# ---- item 4: a fully refused apply creates nothing --------------------------

def test_fully_refused_apply_creates_nothing_and_keeps_the_run_id(mem, tmp_path):
    run_dir = _merge_into_existing(mem, tmp_path)
    res = apply_mod.apply_run(run_dir, yes=True, log=quiet)
    assert res["executed"] == 0 and res["refused"] == 1
    assert not (run_dir / "snapshot-before").exists()
    assert not (mem / ".retired").exists()
    _edit_item(run_dir, merged_slug="brand_new_merged")
    again = apply_mod.apply_run(run_dir, yes=True, log=quiet)
    assert again["executed"] == 1


# ---- item 5: undo scope matches the documented scope ------------------------

def test_undo_restores_exactly_the_snapshotted_md_files(mem, tmp_path):
    (mem / "settings.json").write_text("{}", encoding="utf-8")
    run_dir = _propose(mem, tmp_path)
    _mark(run_dir)
    before = {p.name: p.read_bytes() for p in mem.glob("*.md")}
    apply_mod.apply_run(run_dir, yes=True, log=quiet)
    (mem / "settings.json").write_text('{"changed": true}', encoding="utf-8")
    apply_mod.undo_run(run_dir, yes=True, log=quiet)
    assert {p.name: p.read_bytes() for p in mem.glob("*.md")} == before
    # outside the documented scope: left as the user has it
    assert (mem / "settings.json").read_text() == '{"changed": true}'
    assert (mem / ".retired" / "t").is_dir()
    assert sorted(p.name for p in (run_dir / "snapshot-before").iterdir()) == sorted(before)


# ---- later 1: a missing source is not "executed" ----------------------------

def test_missing_source_is_refused_not_counted_as_executed(mem, tmp_path):
    run_dir = _propose(mem, tmp_path)
    _mark(run_dir)
    payload = json.loads((run_dir / "proposal.json").read_text())
    (mem / f"{payload['items'][0]['files'][0]}.md").unlink()
    res = apply_mod.apply_run(run_dir, yes=True, log=quiet)
    assert res["executed"] == 0 and res["refused"] == 1
    assert not (mem / ".retired").exists()


# ---- later 3: distinct session ids keep distinct seen-sets ------------------

def test_distinct_session_ids_do_not_share_a_seen_file(tmp_path):
    ids = ["a/b", "a?b", "a:b", "a_b"]
    assert len({recall_mod._seen_path(tmp_path, i).name for i in ids}) == len(ids)
    recall_mod.save_seen(tmp_path, "a/b", {"x"})
    assert recall_mod.load_seen(tmp_path, "a?b") == set()
    assert recall_mod.load_seen(tmp_path, "a/b") == {"x"}
