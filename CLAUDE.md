# Project Conventions — music-streaming-user-behavior-analytics

## Context
End-to-end analytics project on simulated music-streaming data (50K users, ~1.5M events).
Snapshot date 2026-04-01; observation window 2026-03-02..2026-04-01; churn label = no
listening activity in 14d post-snapshot; conversion label = paid start within 30d.
Notebooks 01-05 are complete. Existing baselines to respect: 14d churn 47.6%, 30d paid
conversion 17.4%, logistic regression churn model ROC-AUC 0.788.

## Architecture rules (non-negotiable)
- All reusable logic lives in `src/` as typed, docstringed modules. Notebooks are thin
  narrative layers: load data via `src.data_loader.connect()` or `load_table()`, call
  `src/` functions, visualize, and interpret. No statistical formula or model pipeline
  may be defined inline in a notebook if it is used more than once.
- SQL stays the single source of truth: any new cohort/table definition goes in `sql/`
  and is executed through the loader, mirroring `sql/build_user_feature_table.sql`.
- Every module gets pytest unit tests under `tests/` (create the directory on first
  use, add `pytest` to requirements.txt, include a minimal `pytest.ini` or
  `[tool.pytest.ini_options]`). Tests must not require the full dataset: use small
  synthetic fixtures.
- Determinism: every stochastic step takes an explicit `random_state`/seed, with the
  project default `RANDOM_SEED = 42` defined once in `src/config.py` (create it;
  also centralize snapshot/window date constants there and refactor call sites
  opportunistically, without changing notebook 01-05 outputs).
- Style: match the existing codebase — `from __future__ import annotations`, type
  hints, module docstring with usage example, guard-clause QA checks that raise with
  actionable messages (see scripts/build_user_feature_table.py for the house style).

## Notebook conventions
- Numbered continuation: 06, 07, 08, 09. Filename pattern matches existing ones.
- Each notebook opens with a scope cell (business question, method, data dependencies)
  and closes with a data-grounded "Key Findings" section written for a hiring-manager
  reader: numbers first, caveats explicit, no hype.
- Every chart has a takeaway title (the conclusion, not the axis description).
- Execute the notebook end-to-end before committing:
  `jupyter nbconvert --to notebook --execute --inplace notebooks/<nb>.ipynb`
  A notebook with stale or missing outputs must not be committed.

## Delivery discipline
- One feature = one commit = one push. Conventional commit style, e.g.
  `feat(ab-test): add experiment simulation, SRM check and decision framework`.
  Body: 3-6 bullet summary of what was added and headline numbers.
- Each feature updates README.md: Analysis table row, Project Status checkbox,
  and one new Key Findings bullet if the result is headline-worthy.
- Run `pytest` and the nbconvert execution before every commit; both must pass.
- After pushing, STOP and wait for human review. Do not start the next feature.
- Never commit data files > 50 MB; gzip if needed (house pattern already exists).
