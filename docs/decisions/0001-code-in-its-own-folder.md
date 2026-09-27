# 0001 — Code lives in `metanoia/`, separate from the context documents

**Choice.** The Python project (`pyproject.toml`, `src/`, `configs/`, `suites/`,
`analysis/`, `tests/`, `artifacts/`) is in the `metanoia/` subfolder. The design and
research documents (`PLAN.md`, `OPTIONS.md`, `SCIENTIFIC_*`, `idea.docx`) stay in the
parent folder.

**Reason.** This was the owner's explicit request. It keeps the reference documents
apart from the code that changes often.

**Alternatives.** PLAN.md §4 put the documents at the repository root, next to the code.

**Compatibility impact.** None for the code. Paths in configurations are relative to
`metanoia/`, so commands run from there. Links to the documents use `../`.

**Revisit if** the two need separate version histories, or if they need to live in a
single repository root.
