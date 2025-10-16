# BiasXplainer - Batch & Comparative Analysis

This repository provides BiasGuard Pro — a gender-bias detection and explanation tool. I added batch processing, multi-text input, progress tracking, summary statistics, and a simple comparative analysis dashboard.

Quick features
- Single-text interactive analysis with SHAP explanations
- Batch processing (textarea + file upload: .txt, .csv, .json)
- Progress reporting for batch runs
- Summary statistics (avg bias, class counts, top biased words)
- Simple comparison between two groups (substring filters)
- Export batch results to JSON/CSV (saved in `./export/`)

How to run (development)

1. Create a Python environment and install dependencies from `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Launch the Gradio dashboard (web UI):

```bash
python main.py
```

The app will open and present the interactive dashboard. Use the "Batch & Compare" tab to paste multiple texts (one per line) or upload a file.

Batch input formats
- .txt: newline-separated texts
- .csv: must have a `text` column or the first text-like column will be used
- .json: either a list of strings, or a list of objects each containing a `text` field

Export
- When running a batch, you may provide a save path (e.g., `./export/results.json`) to save results.
- After running a batch, use the Export JSON / Export CSV buttons to save downloadable files under `./export/`.

See `docs/usage.md` for an extended usage guide and troubleshooting.

Running tests

```bash
pytest -q
```

Notes & next steps
- For large batches, consider enabling batched inference and/or background job queue to avoid blocking the UI.
- Real-time streaming progress can be added using a background runner and a polling endpoint.

If you want, I can implement the next item: real-time progress streaming, batched inference, or more tests/documentation. Reply with which item to do next and I'll proceed step-by-step.
