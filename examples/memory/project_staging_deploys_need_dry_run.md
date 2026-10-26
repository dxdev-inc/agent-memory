---
name: project_staging_deploys_need_dry_run
description: "Staging deploys must be rehearsed with a dry run before the real deploy runs."
metadata:
  type: project
---

Every staging deploy gets a dry run first, then the real run.

**Why:** A skipped rehearsal once pushed a bad config to staging and QA lost an afternoon.

**How to apply:** Run the deploy script with its dry-run flag, read what it prints, then run it for real. The full sequence is in [[reference_deploy_checklist]].
