"""agent-memory: one command, five subcommands.

    agent-memory check         --memory-dir DIR
    agent-memory rebuild-index --memory-dir DIR [--areas FILE]
    agent-memory recall "<task hint>" --memory-dir DIR [--sid ID] [-n 5] [--json]
    agent-memory propose       --memory-dir DIR [--runs-dir DIR] [--run-id ID]
    agent-memory apply RUN_DIR [--yes] [--undo]

The memory directory comes from --memory-dir or the AGENT_MEMORY_DIR
environment variable. `propose` only writes a proposal; `apply` is the separate,
explicit, reversible step.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import apply as apply_mod
from . import check as check_mod
from . import propose as propose_mod
from . import recall as recall_mod
from .corpus import load_dir


def _memory_dir(args: argparse.Namespace) -> Path | None:
    spec = args.memory_dir or os.environ.get("AGENT_MEMORY_DIR")
    if not spec:
        print("agent-memory: give --memory-dir DIR or set AGENT_MEMORY_DIR",
              file=sys.stderr)
        return None
    p = Path(spec)
    if not p.is_dir():
        print(f"agent-memory: memory dir not found: {p}", file=sys.stderr)
        return None
    return p


def _cmd_check(args: argparse.Namespace) -> int:
    mem = _memory_dir(args)
    return 2 if mem is None else check_mod.check(mem)


def _cmd_rebuild(args: argparse.Namespace) -> int:
    mem = _memory_dir(args)
    if mem is None:
        return 2
    try:
        return check_mod.rebuild_index(mem, args.areas)
    except (OSError, ValueError) as e:
        print(f"agent-memory: cannot read areas file: {e}", file=sys.stderr)
        return 2


def _cmd_recall(args: argparse.Namespace) -> int:
    mem = _memory_dir(args)
    if mem is None:
        return 2
    memories = load_dir(mem)
    seen_dir = mem / ".recall-seen"
    seen = recall_mod.load_seen(seen_dir, args.sid) if args.sid else set()
    hits = recall_mod.recall(args.hint, memories, n=args.n, seen=seen)
    if args.sid and hits:
        recall_mod.save_seen(seen_dir, args.sid, seen | {h.slug for h in hits})
    if args.json:
        print(json.dumps([h.as_dict() for h in hits], indent=2))
    elif not hits:
        print("no relevant memory for that hint")
    else:
        for h in hits:
            print(f"  {h.score:.2f}  {h.slug}  ({h.age_days}d old)\n"
                  f"        {h.description[:160]}")
    return 0


def _cmd_propose(args: argparse.Namespace) -> int:
    mem = _memory_dir(args)
    if mem is None:
        return 2
    try:
        run_dir, summary = propose_mod.propose(
            mem, runs_dir=args.runs_dir, run_id=args.run_id)
    except propose_mod.ProposeError as e:
        print(f"agent-memory propose: {e}", file=sys.stderr)
        return 1
    print()
    print(f"  {summary['files']} memories read, {summary['items']} proposal(s), "
          f"{summary['applicable']} applicable")
    print(f"  review:  {(run_dir / 'proposal.md').as_posix()}")
    print(f"  then:    agent-memory apply {run_dir.as_posix()}")
    return 0


def _cmd_apply(args: argparse.Namespace) -> int:
    try:
        if args.undo:
            apply_mod.undo_run(Path(args.run_dir), yes=args.yes)
        else:
            apply_mod.apply_run(Path(args.run_dir), yes=args.yes)
    except apply_mod.ApplyError as e:
        print(f"agent-memory apply: {e}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent-memory", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_dir(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--memory-dir", help="memory folder "
                        "(or set AGENT_MEMORY_DIR)")

    c = sub.add_parser("check", help="size caps, index and link health; "
                                     "exit 1 on any problem")
    add_dir(c)
    c.set_defaults(func=_cmd_check)

    r = sub.add_parser("rebuild-index", help="regenerate INDEX.md from "
                                             "frontmatter")
    add_dir(r)
    r.add_argument("--areas", type=Path, help='optional JSON file: {"areas": '
                   '{"Area name": ["keyword", ...]}}; default groups by type')
    r.set_defaults(func=_cmd_rebuild)

    rc = sub.add_parser("recall", help="top memories for a task hint "
                                       "(local lexical, no model call)")
    rc.add_argument("hint")
    add_dir(rc)
    rc.add_argument("-n", type=int, default=recall_mod.DEFAULT_N)
    rc.add_argument("--sid", help="suppress memories already surfaced to this "
                                  "session id, and record the new ones")
    rc.add_argument("--json", action="store_true")
    rc.set_defaults(func=_cmd_recall)

    pr = sub.add_parser("propose", help="nominate near-duplicates into a "
                                        "reviewable proposal (changes nothing)")
    add_dir(pr)
    pr.add_argument("--runs-dir", type=Path,
                    help=f"where proposals go (default ./{propose_mod.DEFAULT_RUNS_DIR})")
    pr.add_argument("--run-id", help="name for this run (default: timestamp)")
    pr.set_defaults(func=_cmd_propose)

    a = sub.add_parser("apply", help="execute a reviewed proposal")
    a.add_argument("run_dir")
    a.add_argument("--yes", action="store_true",
                   help="actually do it (default is a dry run)")
    a.add_argument("--undo", action="store_true",
                   help="restore the memory dir from this run's snapshot")
    a.set_defaults(func=_cmd_apply)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
