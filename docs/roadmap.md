# BiasXplainer — Roadmap (expanded)

This roadmap outlines planned work, priorities and potential milestones for BiasXplainer. Use this as a living document to coordinate development and contributions.

## Short-term (next 1-3 months)

1. Improve documentation and tests
   - Add unit tests for `CounterfactualGenerator` template expansions and replacements.
   - Add examples for SHAP fallback behavior and batch export flows.
2. Packaging & CI
   - Add GitHub Actions for linting (black/isort), tests (pytest) and build checks.
   - Create a minimal PyPI package configuration (check `setup.py`) and publish a beta release.
3. Better PDF/Export support
   - Replace the lightweight PDF stub with a proper PDF renderer (WeasyPrint or ReportLab).

## Medium-term (3-9 months)

1. Model improvements
   - Provide a small fine-tuned model checkpoint trained to detect gendered stereotypes more accurately.
   - Add model calibration utilities and evaluation scripts (precision/recall by subgroup).
2. Explainability & UX
   - Improve SHAP sampling strategy to be faster and provide deterministic seeds for reproducibility.
   - Add a compact REST API wrapper (FastAPI) for programmatic access with rate limiting for batch jobs.
3. Evaluation & datasets
   - Collect a small labeled dataset for bias detection (ethical, consented sources) to benchmark model performance.

## Long-term (9-18 months)

1. Integration & plugins
   - Build a plugin system to allow domain-specific rule-sets or templates for counterfactuals.
   - Provide browser extensions or editor plugins (VS Code) to highlight biased text in the editor.
2. Governance & community
   - Create contribution guidelines, a Code of Conduct and a roadmap board for public issue prioritization.

## Optional / Nice-to-have

- Multi-lingual support: extend detectors and templates for non-English languages.
- Real-time inference: WebSocket-backed service that highlights text as the user types.
- Active learning loop: allow human reviewers to confirm/correct model predictions and use those labels to refine the model.

## Contributors & Ownership

Primary owner: dyra-12 (repository)
Open to contributors: yes — accept PRs with tests and clear descriptions.

## How to help

- File issues with concrete reproduction steps.
- Provide small PRs: docs, tests, or bugfixes are the most welcome.
- Validate exports and edge-cases with real sample text from target domains.

---

This roadmap is intentionally lightweight — if you want a project board or milestones created in GitHub I can scaffold a GitHub Projects board and open issues for the short-term items.