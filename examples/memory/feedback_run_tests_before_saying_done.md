---
name: feedback_run_tests_before_saying_done
description: "Run the test suite and read the result before telling the user a change is done or fixed."
metadata:
  type: feedback
---

Never say a change is done, fixed or working until the tests have actually run and passed in this session.

**Why:** Twice a change was reported as done and the suite was red when the user ran it. "Should work" is not a result.

**How to apply:** Run the relevant tests, read the output, and say which ones ran. If something failed that you did not cause, say so plainly instead of hiding it. See also [[project_staging_deploys_need_dry_run]].
