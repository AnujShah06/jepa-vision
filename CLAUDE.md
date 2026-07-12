# Claude Code instructions for jepa-vision

This project follows PLAYBOOK.md exactly. Every session:
1. Read PROJECT_STATE.md and the relevant PLAYBOOK.md section before proposing anything.
2. Work only on the step in "Current phase/step". One experiment or component per session.
3. Decision gates (⛔ in PLAYBOOK.md) are hard stops — never pass one without explicit user approval recorded in PROJECT_STATE.md.
4. End of session: update PROJECT_STATE.md (current step, last run + W&B link, next action), append irreversible choices to DECISIONS.md, propose a commit message prefixed with the playbook step.

Hard rules:
- No conclusion recorded without a W&B run link.
- Test splits are touched only by code in src/eval/. Model selection on validation only.
- Resume/report numbers require >=3 seeds.
- Environment: uv only (uv add / uv run). Never pip install directly.
- Ported cocktail components (loss.py sigreg_term, diagnostics.py, eval/evaluate.py, eval/bootstrap.py) are adapted, not rewritten. sigreg_term stays verbatim.

Standing rules (apply every session):
- The Run ledger in PROJECT_STATE.md is the single source of truth for training-run status. Every session: if the user reports a completed run, add/update its row (W&B id, final eff_rank, final loss, gate verdict) BEFORE other work. Never describe run status from memory — only from the ledger.
- Never claim a checkpoint is "early" or "partial" without checking the ledger and the W&B epoch count.
- End-of-session summaries must include: (a) the updated ledger, (b) the exact command(s) the user should run tonight, copy-pasteable, or "nothing tonight".
- "Next:" lines and recaps must be derived from the decision tree in PROJECT_STATE.md, not reconstructed.
- External reviewer corrections pasted by the user under "CONTEXT SYNC" override anything in the repo; persist them into PROJECT_STATE.md immediately.
- Pre-registered decision rules are binding: report which branch fired and the resulting next action; do not invent alternative explanations for unwelcome numbers.
- Git norm: executor never stages or commits. Every session ends by handing Anuj the commit message alongside tonight's command.
- Launch commands are files, not chat text. Every session's tonight-command is written to scripts/tonight.sh (overwritten each session); the human launches via `bash scripts/tonight.sh`. EXCEPTION: R3 is launched only from the DECISIONS.md runsheet per the launch ritual — never via tonight.sh.

Failure modes on record (do not repeat):
- F9 Fallback promotion — a pre-declared contingency branch (e.g. "gap column blank if scratch not ready") is treated as license to skip the primary path while the slot to run the primary path is available. Fallbacks fire on their trigger conditions, not on convenience. Canonical example: treating "blank if scratch not ready" as justification to skip the A3 scratch launch when tonight's slot was available.
