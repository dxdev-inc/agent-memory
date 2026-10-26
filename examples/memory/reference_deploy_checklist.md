---
name: reference_deploy_checklist
description: "The ordered deploy steps: build, dry run, deploy, smoke check, announce."
metadata:
  type: reference
---

1. Build from a clean checkout.
2. Dry run first.
3. Deploy.
4. Smoke check the three main pages.
5. Announce in the team channel.

**Why:** The order matters. Announcing early once sent people to a page that was down.

**How to apply:** Go in order and stop at the first failure. The policy behind step 2 is in [[project_staging_deploys_need_dry_run]].
