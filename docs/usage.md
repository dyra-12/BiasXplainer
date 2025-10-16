# BiasXplainer — Usage Guide

This document covers how to use the BiasGuard Pro application for single analyses, batch processing, exports, and background jobs.

Requirements

- Python 3.10+ recommended
- Install dependencies from `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Launching the GUI (Gradio)

```bash
python main.py
```

The Gradio dashboard will open. Key areas:

- Input Text: analyze a single text with SHAP explanations and counterfactual suggestions.
- Quick Examples: sample prompts to populate the input box.
- Batch & Compare tab: paste multiple texts (one per line) or upload a file, then start a background batch job and refresh status.

Batch input formats

- Plain text (.txt): newline-separated texts.
- CSV (.csv): include a `text` column. If absent, the code will use the first available column per row.
- JSON (.json): either a list of strings, or a list of objects with a `text` field.

Background batch jobs

- Click "Start Background Batch" to create a background job. A Job ID is returned.
- Click "Refresh Job Status" and paste the Job ID to poll the job progress.
- When finished, the job record contains `results`, `summary`, and optional `comparison` (if you supplied group filters).

Exports

- When starting a background job you can provide a save path (e.g. `./export/results.json` or `./export/results.csv`) and the worker will attempt to write the results.
- After a run you can also use the Export JSON / Export CSV buttons to save the last-run results; files are written under `./export/` by default.

CLI-style batch example (programmatic)

You can import and use the dashboard classes in scripts. Example:

```python
from main import BiasGuardDashboard

# Construct dashboard (this will initialize models)
d = BiasGuardDashboard()

texts = [
  "Women should be nurses because they are compassionate.",
  "Men are naturally better at engineering roles.",
  "This is a neutral sentence."
]

# Run a blocking batch (not background)
results = d.analyze_batch(texts)
summary = d.summarize_batch(results)
print(summary)
```

Testing

Run unit tests with:

```bash
pytest -q
```

Notes and best practices

- For very large batches, prefer running scoring-only batches (you can adapt the code to call `detector.predict_batch_batched` directly) and run SHAP explanations only on a subset.
- The current background job runner is in-memory and suitable for single-machine development. For production, use a queue (e.g., Redis + RQ/Celery) and persistent job state.
- If you run into thread-safety issues with heavy models or SHAP in background threads, run the worker in a separate process or use a queue that launches worker processes.

Troubleshooting

- "Model load errors": ensure model files exist under `./models` or `model_path` points to a valid Hugging Face model.
- "Slow batches": reduce SHAP usage or increase batch_size in `predict_batch_batched`.

Contact

If you need additional features (streaming progress via websocket, Celery integration, or hosted deployment), open an issue or request the feature and I can implement it next.
