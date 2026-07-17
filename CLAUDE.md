# Project Conventions — music-streaming-user-behavior-analytics

## Context
Complete end-to-end analytics project on simulated music-streaming data (50K
users, ~1.5M events): notebooks 01-10, nine `src/` modules, 75 unit tests,
CI. Snapshot date 2026-04-01; observation window 2026-03-02..2026-04-01;
churn label = no listening activity in 14d post-snapshot. The project is in
maintenance mode: changes are fixes, hardening, or explicitly requested
extensions — not new roadmap items.

## Baseline invariants (locked by tests/test_baseline_invariants.py)
Never "update the expected number" — a shift means a regression:
- feature_table: 50,000 users; 14d churn rate 0.475580; 30d conversion
  rate 0.173920.
- Churn logistic baseline (notebooks 05/07 parity gate): ROC-AUC 0.787615,
  precision 0.670393, recall 0.781749 at the 0.5 threshold.
- Experiment dataset (notebook 06): measured ATE +2.58pp
  (0.025761), injected ground-truth mean +2.2252pp.
- Conversion baseline (notebook 10, in src/conversion_modeling.py
  docstring): LR PR-AUC 0.7274, ROC-AUC 0.8939, Brier 0.0994.

## CI (must stay green)
- Every push runs `ruff check .` + bare `pytest` on Python 3.11/3.12; both
  must pass before any push. Verify locally the way CI runs them (bare
  `pytest`, not only `python -m pytest`).
- The manual `notebooks` workflow executes 01-10 in order; run it after any
  change that touches notebook inputs.
- Dependencies: `requirements.txt` is the loose declaration;
  `requirements.lock` (pip freeze from a venv where the full suite passes)
  is what CI and `make setup` install. Regenerate the lock when
  requirements.txt changes.

## Architecture rules (non-negotiable)
- All reusable logic lives in `src/` as typed, docstringed modules. Notebooks
  are thin narrative layers: load data via `src.data_loader`, call `src/`
  functions, visualize, and interpret. No statistical formula or model
  pipeline may be defined inline in a notebook if it is used more than once.
- SQL stays the single source of truth: cohort/table definitions live in
  `sql/` and are executed through the loader.
- Every module keeps pytest coverage under `tests/`; tests never require the
  full dataset (small synthetic fixtures; SQL is tested on synthetic sqlite
  event logs).
- Determinism: every stochastic step takes an explicit seed from
  `src/config.py` (`RANDOM_SEED = 42`); date constants are centralized there.
- Style: `from __future__ import annotations`, type hints, module docstring
  with usage example, guard-clause QA checks that raise with actionable
  messages. `ruff check` and `ruff format` are the arbiters (config in
  pyproject.toml; notebooks are excluded from formatting).

## Notebook conventions
- Numbered 01-10; filenames match the existing pattern.
- Each notebook opens with a scope cell (business question, method, data
  dependencies) and closes with a data-grounded "Key Findings" section:
  numbers first, caveats explicit, no hype.
- Every chart has a takeaway title (the conclusion, not the axis description).
- A notebook is committed only fully executed with embedded outputs:
  `jupyter nbconvert --to notebook --execute --inplace notebooks/<nb>.ipynb`
- No brand names, no coursework language, no TODO markers anywhere. When
  scanning notebooks for banned words, check cell sources and text outputs —
  base64 image payloads can false-positive.

## Delivery discipline
- One change = one commit = one push. Conventional commit style
  (`fix(scope): ...`, `chore(scope): ...`); body = 3-6 bullets with headline
  numbers.
- Run `ruff check .`, bare `pytest`, and (when notebooks changed) the
  affected notebook executions before every commit.
- Changes that touch README-quoted numbers must update the README in the
  same commit — and if a locked invariant moved, stop and fix the
  regression instead.
- After pushing, STOP and wait for human review. Do not start follow-up
  work unprompted.
- Never commit data files > 50 MB; gzip if needed (house pattern exists).
