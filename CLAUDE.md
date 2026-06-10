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
