---
name: feedback_no_force_push_to_main
description: "Never force push to main or any shared branch; use a new commit or a revert instead."
metadata:
  type: feedback
---

Never force push to main or to any branch other people pull from.

**Why:** A force push rewrote shared history once and teammates lost a day recovering their work.

**How to apply:** Fix mistakes with a new commit or a revert. If a rewrite really is needed, ask first. The agreed branch rules are in [[reference_branch_policy]].
