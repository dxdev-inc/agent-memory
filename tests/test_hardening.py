"""Regression tests from the pre-publication review: run ids, runs folder
location, double apply, dry-run preflight, CRLF byte counts, old key name,
quoted frontmatter values with comments."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from agent_memory import apply as apply_mod
from agent_memory import check as check_mod
from agent_memory import propose as propose_mod
from agent_memory.cli import main
from agent_memory.corpus import parse_frontmatter

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


# ---- item 2: run id --------------------------------------------------------

@pytest.mark.parametrize("bad", ["../escaped", "a/b", "a\\b", "..", "a..", "..a",
                                 "a..b", ".hidden", "trail.", "", "C:x", "a b"])
def test_propose_refuses_unsafe_run_ids(mem, tmp_path, bad):
    with pytest.raises(propose_mod.ProposeError):
        propose_mod.propose(mem, runs_dir=tmp_path / "runs", run_id=bad, log=quiet)
    assert not (tmp_path / "escaped").exists()
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("good", ["demo", "run-1", "run_2", "v1.2", "2026-11-01"])
def test_propose_accepts_sane_run_ids(mem, tmp_path, good):
    assert _propose(mem, tmp_path, good).name == good


def test_cli_propose_run_id_traversal_exits_nonzero(mem, tmp_path, capsys):
    rc = main(["propose", "--memory-dir", str(mem), "--runs-dir",
               str(tmp_path / "runs"), "--run-id", "../escaped"])
    assert rc == 1
    assert "refusing run id" in capsys.readouterr().err
    assert not (tmp_path / "escaped").exists()


def test_apply_refuses_a_run_dir_with_an_unsafe_name(mem, tmp_path):
    run_dir = _propose(mem, tmp_path)
    bad = tmp_path / "runs" / "a..b"
    shutil.copytree(run_dir, bad)
    with pytest.raises(apply_mod.ApplyError, match="refusing run id"):
        apply_mod.apply_run(bad, yes=True, log=quiet)


# ---- item 3: runs folder inside the memory folder --------------------------

def test_propose_refuses_runs_dir_inside_memory(mem):
    with pytest.raises(propose_mod.ProposeError, match="inside the memory folder"):
        propose_mod.propose(mem, runs_dir=mem / ".runs", run_id="t", log=quiet)
    assert not (mem / ".runs").exists()


def test_cli_propose_default_runs_dir_from_inside_memory_refuses(mem, monkeypatch, capsys):
    monkeypatch.chdir(mem)
    assert main(["propose", "--memory-dir", ".", "--run-id", "t"]) == 1
    assert "inside the memory folder" in capsys.readouterr().err


def test_apply_refuses_when_run_dir_is_inside_memory(mem, tmp_path):
    run_dir = _propose(mem, tmp_path)
    inside = mem / ".runs" / "t"
    shutil.copytree(run_dir, inside)
    _mark(inside)
    before = sorted(p.name for p in mem.rglob("*"))
    with pytest.raises(apply_mod.ApplyError, match="inside the memory folder"):
        apply_mod.apply_run(inside, yes=True, log=quiet)
    assert not (inside / "snapshot-before").exists()
    assert sorted(p.name for p in mem.rglob("*")) == before


# ---- item 4: applying the same run twice -----------------------------------

def test_second_apply_is_refused_and_manifest_survives(mem, tmp_path):
    run_dir = _propose(mem, tmp_path)
    _mark(run_dir)
    apply_mod.apply_run(run_dir, yes=True, log=quiet)
    manifest = mem / ".retired" / "t" / "MANIFEST.json"
    original = manifest.read_bytes()
    assert json.loads(original)["entries"]
    with pytest.raises(apply_mod.ApplyError, match="already applied"):
        apply_mod.apply_run(run_dir, yes=True, log=quiet)
    assert manifest.read_bytes() == original


def test_undo_still_works_after_a_refused_second_apply(mem, tmp_path):
    before = {p.name: p.read_bytes() for p in mem.glob("*.md")}
    run_dir = _propose(mem, tmp_path)
    _mark(run_dir)
    apply_mod.apply_run(run_dir, yes=True, log=quiet)
    with pytest.raises(apply_mod.ApplyError):
        apply_mod.apply_run(run_dir, yes=True, log=quiet)
    apply_mod.undo_run(run_dir, yes=True, log=quiet)
    assert {p.name: p.read_bytes() for p in mem.glob("*.md")} == before


def test_cli_second_apply_exits_nonzero(mem, tmp_path, capsys):
    run_dir = _propose(mem, tmp_path)
    _mark(run_dir)
    assert main(["apply", str(run_dir), "--yes"]) == 0
    assert main(["apply", str(run_dir), "--yes"]) == 1
    assert "already applied" in capsys.readouterr().err


# ---- item 5: dry run predicts the merge-target refusal ---------------------

def _merge_into_existing(mem, tmp_path):
    run_dir = _propose(mem, tmp_path)
    pj = run_dir / "proposal.json"
    payload = json.loads(pj.read_text())
    other = next(p.stem for p in mem.glob("*.md")
                 if p.stem not in payload["items"][0]["files"]
                 and p.name not in ("MEMORY.md", "INDEX.md"))
    payload["items"][0].update(action="merge", merged_slug=other,
                               merged_body="merged body", applicable=True)
    pj.write_text(json.dumps(payload), encoding="utf-8")
    _mark(run_dir)
    return run_dir


def test_dry_run_predicts_the_merge_target_refusal(mem, tmp_path):
    run_dir = _merge_into_existing(mem, tmp_path)
    lines: list[str] = []
    dry = apply_mod.apply_run(run_dir, yes=False, log=lines.append)
    assert dry["refused"] == 1 and dry["would_execute"] == 0
    assert any("WOULD REFUSE" in line for line in lines)
    assert not any(line.startswith("WOULD merge") for line in lines)
    real = apply_mod.apply_run(run_dir, yes=True, log=quiet)
    assert real["refused"] == dry["refused"] and real["executed"] == 0


# ---- item 6: CRLF byte count -----------------------------------------------

def test_check_counts_crlf_bytes_raw(tmp_path, capsys):
    mem = tmp_path / "m"
    mem.mkdir()
    line = "x" * 126 + "\r\n"            # 128 bytes per line
    data = (line * 200).encode()
    assert len(data) == 25600
    (mem / "MEMORY.md").write_bytes(data + b"y" * 200)  # 25,800 bytes
    assert check_mod.check(mem) == 1
    out = capsys.readouterr().out
    assert "25800 bytes" in out and "OVER HARD LIMIT" in out


# ---- later 1: old key name -------------------------------------------------

def test_apply_reads_the_old_applyable_key(mem, tmp_path):
    run_dir = _propose(mem, tmp_path)
    pj = run_dir / "proposal.json"
    payload = json.loads(pj.read_text())
    for it in payload["items"]:
        it["applyable"] = it.pop("applicable")
    pj.write_text(json.dumps(payload), encoding="utf-8")
    _mark(run_dir)
    execute, _, blocked = apply_mod.plan(run_dir)
    assert len(execute) == 1 and not blocked


# ---- later 2: quoted value followed by a comment ---------------------------

@pytest.mark.parametrize("raw,want", [
    ('"foo" # note', "foo"),
    ("'foo' # note", "foo"),
    ('"foo"', "foo"),
    ('"a # b"', "a # b"),
    ("foo # note", "foo"),
    ('"say "hi" now"', 'say "hi" now'),
])
def test_frontmatter_quoted_value_with_trailing_comment(raw, want):
    fm, _ = parse_frontmatter(f"---\nname: n\ndescription: {raw}\n---\nbody\n")
    assert fm["description"] == want
