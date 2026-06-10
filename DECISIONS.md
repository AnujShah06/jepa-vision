# DECISIONS.md — append-only log of irreversible choices

Format: `[Step] Decision — one-line rationale`

---

[0.3] **Three seeds for multi-seed protocol** — vision SSL community standard (three seeds is the norm for STL-10-scale experiments); five seeds was the cocktail project's choice for a faster run. Logged as a deliberate reduction. Any resume/report number still requires ≥3 seeds.

[0.3] **uv as the sole package manager** — reproducible lockfile, fast installs, no pip-install side channels.

[0.3] **SIGReg retained as the anti-collapse regularizer** — ported from cocktail-JEPA Phase-2 fix #18. EMA is also retained (not dropped as LeJEPA suggests) because the energy function is built on the EMA target encoder's latents; dropping EMA removes the thing the energy estimator is built on.

[0.3] **sigreg_term kept verbatim** — the Epps-Pulley characteristic-function test is mathematically settled; no adaptation needed for the vision domain. Only jepa_loss (its caller) is lightly adapted to remove recipe-specific terminology.
