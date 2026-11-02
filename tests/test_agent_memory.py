"""Tests run against examples/memory, copied to tmp_path before any mutation."""
from __future__ import annotations

import filecmp
import json
import shutil
from pathlib import Path

import pytest

from agent_memory import apply as apply_mod
from agent_memory import check as check_mod
from agent_memory import propose as propose_mod
from agent_memory import recall as recall_mod
from agent_memory.cli import main
from agent_memory.corpus import load_dir

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "memory"
DUP_A = "feedback_run_tests_before_saying_done"
DUP_B = "feedback_test_before_declaring_done"


@pytest.fixture
def mem(tmp_path: Path) -> Path:
    dest = tmp_path / "memory"
    shutil.copytree(EXAMPLE, dest)
    return dest


@pytest.fixture
def indexed(mem: Path) -> Path:
    check_mod.rebuild_index(mem)
    return mem


def _snapshot(d: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(d.glob("*.md"))}


# ---- check ----------------------------------------------------------------

def test_example_without_index_fails_then_passes_after_rebuild(mem, capsys):
    assert check_mod.check(mem) == 1
    assert "INDEX.md missing" in capsys.readouterr().out
    assert check_mod.rebuild_index(mem) == 0
    assert check_mod.check(mem) == 0


def test_unresolved_wikilink_is_reported_but_not_fatal(indexed, capsys):
    assert check_mod.check(indexed) == 0
    assert "reference_branch_policy" in capsys.readouterr().out


def test_dangling_memory_md_link_fails(indexed, capsys):
    with (indexed / "MEMORY.md").open("a", encoding="utf-8") as f:
        f.write("- [Gone](does_not_exist.md): missing target\n")
    assert check_mod.check(indexed) == 1
    assert "does_not_exist.md" in capsys.readouterr().out


def test_dangling_index_link_and_orphan_fail(indexed, capsys):
    with (indexed / "INDEX.md").open("a", encoding="utf-8") as f:
        f.write("- [Gone](ghost.md)\n")
    (indexed / "new_orphan.md").write_text(
        "---\nname: new_orphan\ndescription: not indexed\n---\nbody\n",
        encoding="utf-8")
    assert check_mod.check(indexed) == 1
    out = capsys.readouterr().out
    assert "ghost.md" in out and "new_orphan.md" in out


def test_missing_description_fails(indexed, capsys):
    (indexed / "no_desc.md").write_text("---\nname: no_desc\n---\nbody\n",
                                        encoding="utf-8")
    check_mod.rebuild_index(indexed)
    assert check_mod.check(indexed) == 1
    assert "missing name/description" in capsys.readouterr().out


def _write_index_only(tmp_path, body: str) -> Path:
    (tmp_path / "MEMORY.md").write_text(body, encoding="utf-8")
    (tmp_path / "INDEX.md").write_text("", encoding="utf-8")
    return tmp_path


def test_past_limit_by_lines():
    text = "\n".join(f"line{i}" for i in range(1, 11))
    assert [n for n, _ in check_mod.past_limit(text, 7, 10_000)] == [8, 9, 10]


def test_past_limit_by_bytes_drops_whole_line_that_does_not_fit():
    assert check_mod.past_limit("aaaa\nbbbb\ncccc", 100, 9) == [(3, "cccc")]


def test_within_limits_returns_nothing():
    assert check_mod.past_limit("a\nb\nc", 10, 100) == []


def test_check_over_hard_limit_lists_dropped_entries(tmp_path, capsys):
    body = "\n".join(f"- [entry {i}](e{i}.md): hook"
                     for i in range(1, check_mod.HARD_LINES + 6))
    assert check_mod.check(_write_index_only(tmp_path, body)) == 1
    out = capsys.readouterr().out
    assert "NOT LOADED at session start" in out
    assert f"line {check_mod.HARD_LINES + 1}:" in out
    assert "entry 1](" not in out  # loaded entries are not listed


def test_check_over_target_lists_entries_at_risk(tmp_path, capsys):
    body = "\n".join(f"- [entry {i}](e{i}.md): hook"
                     for i in range(1, check_mod.TARGET_LINES + 4))
    assert check_mod.check(_write_index_only(tmp_path, body)) == 1
    out = capsys.readouterr().out
    assert "past the target" in out
    assert f"line {check_mod.TARGET_LINES + 1}:" in out


def test_check_byte_cap(tmp_path, capsys):
    body = "x" * (check_mod.HARD_BYTES + 10)
    assert check_mod.check(_write_index_only(tmp_path, body)) == 1
    assert "OVER HARD LIMIT" in capsys.readouterr().out


# ---- rebuild-index --------------------------------------------------------

def test_rebuild_index_groups_by_type_and_lists_every_file(mem):
    check_mod.rebuild_index(mem)
    text = (mem / "INDEX.md").read_text(encoding="utf-8")
    assert "## Working rules and feedback  (3)" in text
    assert "## About the user  (2)" in text
    for f in mem.glob("*.md"):
        if f.name not in ("MEMORY.md", "INDEX.md"):
            assert f"]({f.name})" in text


def test_rebuild_index_uses_optional_areas_file(mem, tmp_path):
    areas = tmp_path / "areas.json"
    areas.write_text(json.dumps({"areas": {"Deploys": ["deploy"]}}),
                     encoding="utf-8")
    check_mod.rebuild_index(mem, areas)
    text = (mem / "INDEX.md").read_text(encoding="utf-8")
    assert "## Deploys  (2)" in text
    assert "## Other" in text


# ---- recall ---------------------------------------------------------------

def test_recall_finds_the_right_file(mem):
    hits = recall_mod.recall("how do I deploy to staging", load_dir(mem))
    assert hits[0].slug in ("project_staging_deploys_need_dry_run",
                            "reference_deploy_checklist")
    hits = recall_mod.recall("how long should my replies be", load_dir(mem))
    assert hits[0].slug == "user_prefers_short_replies"


def test_recall_returns_nothing_for_an_unrelated_hint(mem):
    assert recall_mod.recall("quarterly tax filing for the llc", load_dir(mem)) == []


def test_recall_seen_set_suppresses_repeats(mem):
    mems = load_dir(mem)
    first = recall_mod.recall("run tests before done", mems, n=1)
    again = recall_mod.recall("run tests before done", mems, n=5,
                              seen={first[0].slug})
    assert first[0].slug not in {h.slug for h in again}


def test_seen_set_roundtrip(tmp_path):
    assert recall_mod.load_seen(tmp_path, "s1") == set()
    recall_mod.save_seen(tmp_path, "s1", {"a", "b"})
    assert recall_mod.load_seen(tmp_path, "s1") == {"a", "b"}
    assert recall_mod.load_seen(tmp_path, "other") == set()


def test_recall_cli_json(mem, capsys):
    rc = main(["recall", "never force push to the shared branch",
               "--memory-dir", str(mem), "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out[0]["slug"] == "feedback_no_force_push_to_main"


# ---- propose --------------------------------------------------------------

def test_propose_finds_the_duplicate_pair_and_only_that(mem, tmp_path):
    run_dir, summary = propose_mod.propose(
        mem, runs_dir=tmp_path / "runs", run_id="t", log=lambda *_: None)
    items = json.loads((run_dir / "proposal.json").read_text())["items"]
    assert summary["items"] == 1
    assert items[0]["action"] == "retire"
    assert {items[0]["keep"], *items[0]["files"]} == {DUP_A, DUP_B}
    assert items[0]["keep"] == DUP_A          # the larger file is kept
    assert "decision: hold" in (run_dir / "proposal.md").read_text()


def test_propose_changes_nothing(mem, tmp_path):
    before = _snapshot(mem)
    propose_mod.propose(mem, runs_dir=tmp_path / "runs", run_id="t",
                        log=lambda *_: None)
    assert _snapshot(mem) == before


# ---- apply and undo -------------------------------------------------------

def _run(mem, tmp_path) -> Path:
    run_dir, _ = propose_mod.propose(mem, runs_dir=tmp_path / "runs",
                                     run_id="t", log=lambda *_: None)
    return run_dir


def _mark_apply(run_dir: Path) -> None:
    md = run_dir / "proposal.md"
    md.write_text(md.read_text().replace("decision: hold", "decision: apply"),
                  encoding="utf-8")


def test_apply_with_no_marks_touches_nothing(mem, tmp_path):
    run_dir = _run(mem, tmp_path)
    before = _snapshot(mem)
    res = apply_mod.apply_run(run_dir, yes=True, log=lambda *_: None)
    assert res["executed"] == 0
    assert _snapshot(mem) == before


def test_apply_is_a_dry_run_without_yes(mem, tmp_path):
    run_dir = _run(mem, tmp_path)
    _mark_apply(run_dir)
    before = _snapshot(mem)
    res = apply_mod.apply_run(run_dir, log=lambda *_: None)
    assert res["dry_run"] and res["would_execute"] == 1
    assert _snapshot(mem) == before


def test_apply_then_undo_restores_byte_for_byte(mem, tmp_path):
    before = _snapshot(mem)
    run_dir = _run(mem, tmp_path)
    _mark_apply(run_dir)
    res = apply_mod.apply_run(run_dir, yes=True, log=lambda *_: None)
    assert res["executed"] == 1
    assert not (mem / f"{DUP_B}.md").exists()
    assert (mem / ".retired" / "t" / f"{DUP_B}.md").exists()
    assert (mem / ".retired" / "t" / "MANIFEST.json").exists()
    assert _snapshot(mem) != before

    apply_mod.undo_run(run_dir, yes=True, log=lambda *_: None)
    assert _snapshot(mem) == before
    for name in before:
        assert filecmp.cmp(EXAMPLE / name, mem / name, shallow=False)


def test_undo_dry_run_changes_nothing(mem, tmp_path):
    run_dir = _run(mem, tmp_path)
    _mark_apply(run_dir)
    apply_mod.apply_run(run_dir, yes=True, log=lambda *_: None)
    after = _snapshot(mem)
    apply_mod.undo_run(run_dir, log=lambda *_: None)
    assert _snapshot(mem) == after


def test_undo_without_snapshot_errors(mem, tmp_path):
    with pytest.raises(apply_mod.ApplyError):
        apply_mod.undo_run(_run(mem, tmp_path), yes=True, log=lambda *_: None)


def test_apply_module_has_no_delete_calls():
    src = Path(apply_mod.__file__).read_text(encoding="utf-8")
    assert "rmtree" not in src
    # the only unlink is the one inside undo_run
    assert src.count(".unlink(") == 1
    assert src.index(".unlink(") > src.index("def undo_run")


def test_merge_item_writes_merged_file_and_undoes(mem, tmp_path):
    run_dir = _run(mem, tmp_path)
    pj = run_dir / "proposal.json"
    payload = json.loads(pj.read_text())
    payload["items"][0].update(
        action="merge", files=[DUP_A, DUP_B], merged_slug="feedback_tests_first",
        merged_description="Run tests before saying done.",
        merged_body="Run the tests first.\n\n**Why:** red suites.\n")
    pj.write_text(json.dumps(payload), encoding="utf-8")
    _mark_apply(run_dir)
    before = _snapshot(mem)
    apply_mod.apply_run(run_dir, yes=True, log=lambda *_: None)
    assert (mem / "feedback_tests_first.md").exists()
    assert not (mem / f"{DUP_A}.md").exists()
    apply_mod.undo_run(run_dir, yes=True, log=lambda *_: None)
    assert _snapshot(mem) == before


# ---- cli ------------------------------------------------------------------

def test_cli_requires_a_memory_dir(monkeypatch, capsys):
    monkeypatch.delenv("AGENT_MEMORY_DIR", raising=False)
    assert main(["check"]) == 2
    assert "AGENT_MEMORY_DIR" in capsys.readouterr().err


def test_cli_reads_memory_dir_from_env(mem, monkeypatch):
    monkeypatch.setenv("AGENT_MEMORY_DIR", str(mem))
    assert main(["rebuild-index"]) == 0
    assert main(["check"]) == 0


def test_cli_full_flow(mem, tmp_path, capsys):
    runs = tmp_path / "runs"
    assert main(["propose", "--memory-dir", str(mem), "--runs-dir", str(runs),
                 "--run-id", "x"]) == 0
    md = runs / "x" / "proposal.md"
    md.write_text(md.read_text().replace("decision: hold", "decision: apply"),
                  encoding="utf-8")
    before = _snapshot(mem)
    assert main(["apply", str(runs / "x"), "--yes"]) == 0
    assert main(["apply", str(runs / "x"), "--undo", "--yes"]) == 0
    assert _snapshot(mem) == before


# ---- review fixes ---------------------------------------------------------


def _edit_item(run_dir: Path, **fields) -> None:
    pj = run_dir / "proposal.json"
    payload = json.loads(pj.read_text())
    payload["items"][0].update(fields)
    pj.write_text(json.dumps(payload), encoding="utf-8")


def test_slug_path_traversal_is_refused_and_nothing_moves(mem, tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("not a memory\n", encoding="utf-8")
    run_dir = _run(mem, tmp_path)
    _edit_item(run_dir, action="retire", files=["../outside"])
    _mark_apply(run_dir)
    before = _snapshot(mem)
    with pytest.raises(apply_mod.ApplyError):
        apply_mod.apply_run(run_dir, yes=True, log=lambda *_: None)
    assert outside.read_text() == "not a memory\n"
    assert _snapshot(mem) == before
    assert not (run_dir / apply_mod.SNAPSHOT).exists()


@pytest.mark.parametrize("slug", ["..", "a/b", "a\\b", ".retired", "C:evil", ""])
def test_unsafe_slugs_are_rejected(mem, slug):
    with pytest.raises(apply_mod.ApplyError):
        apply_mod.check_slug(mem, slug)


def test_unsafe_merged_slug_is_refused(mem, tmp_path):
    run_dir = _run(mem, tmp_path)
    _edit_item(run_dir, action="merge", files=[DUP_A, DUP_B],
               merged_slug="../escaped", merged_body="x\n")
    _mark_apply(run_dir)
    with pytest.raises(apply_mod.ApplyError):
        apply_mod.apply_run(run_dir, yes=True, log=lambda *_: None)
    assert not (tmp_path / "escaped.md").exists()


def test_merge_never_overwrites_an_unrelated_existing_memory(mem, tmp_path):
    keep = next(p for p in sorted(mem.glob("*.md"))
                if p.stem not in (DUP_A, DUP_B, "MEMORY", "INDEX"))
    original = keep.read_bytes()
    run_dir = _run(mem, tmp_path)
    _edit_item(run_dir, action="merge", files=[DUP_A, DUP_B],
               merged_slug=keep.stem, merged_body="Replacement.\n")
    _mark_apply(run_dir)
    res = apply_mod.apply_run(run_dir, yes=True, log=lambda *_: None)
    assert keep.read_bytes() == original
    assert (mem / f"{DUP_A}.md").exists() and (mem / f"{DUP_B}.md").exists()
    assert res["executed"] == 0 and res["refused"] == 1


def test_a_run_id_that_was_applied_cannot_be_reused(mem, tmp_path):
    run_dir = _run(mem, tmp_path)
    _mark_apply(run_dir)
    apply_mod.apply_run(run_dir, yes=True, log=lambda *_: None)
    with pytest.raises(propose_mod.ProposeError):
        _run(mem, tmp_path)
    # the snapshot is still the one taken before the apply, so undo is still right
    apply_mod.undo_run(run_dir, yes=True, log=lambda *_: None)
    assert (mem / f"{DUP_B}.md").exists()


def test_reusing_an_unapplied_run_id_is_fine(mem, tmp_path):
    _run(mem, tmp_path)
    _run(mem, tmp_path)


def test_cli_propose_reports_a_reused_run_id(mem, tmp_path, capsys):
    run_dir = _run(mem, tmp_path)
    _mark_apply(run_dir)
    apply_mod.apply_run(run_dir, yes=True, log=lambda *_: None)
    rc = main(["propose", "--memory-dir", str(mem), "--runs-dir",
               str(tmp_path / "runs"), "--run-id", "t"])
    assert rc == 1 and "already applied" in capsys.readouterr().err


def test_readme_frontmatter_example_parses_literally(tmp_path):
    (tmp_path / "run_tests_first.md").write_text(
        "---\nname: run_tests_first\ndescription: \"One line, with # inside quotes.\"\n"
        "metadata:\n  type: feedback        # user | feedback | project | reference\n---\n\nBody.\n",
        encoding="utf-8")
    (m,) = load_dir(tmp_path)
    assert m.kind == "feedback"
    assert m.description == "One line, with # inside quotes."


def test_final_newline_does_not_add_a_line():
    assert check_mod.count_lines("a\nb\n") == 2
    assert check_mod.count_lines("a\nb") == 2
    assert check_mod.count_lines("") == 0
    assert check_mod.past_limit("a\nb\n", 2, 10_000) == []


def test_file_of_exactly_the_target_lines_is_ok(tmp_path, capsys):
    body = "\n".join(f"- [entry {i}](e{i}.md): hook"
                     for i in range(1, check_mod.TARGET_LINES + 1)) + "\n"
    check_mod.check(_write_index_only(tmp_path, body))
    out = capsys.readouterr().out
    assert f"MEMORY.md: {check_mod.TARGET_LINES} lines" in out
    assert "past the target" not in out


def test_summary_and_json_use_the_correct_spelling(mem, tmp_path, capsys):
    run_dir = _run(mem, tmp_path)
    payload = json.loads((run_dir / "proposal.json").read_text())
    assert payload["summary"]["applicable"] == 1
    assert payload["items"][0]["applicable"] is True
    assert "applyable" not in json.dumps(payload)
    assert "applicable" in (run_dir / "proposal.md").read_text()
