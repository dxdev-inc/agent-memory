---
name: feedback_test_before_declaring_done
description: "Run the tests and read the output before declaring a change done, fixed or working."
metadata:
  type: feedback
---

Do not declare a change done or fixed until the tests have run and passed in this session.

**Why:** A change was reported as done while the test suite was red. "Should work" is not a result.

**How to apply:** Run the relevant tests, read the output, and say which tests ran. If a failure is not yours, say so plainly.
