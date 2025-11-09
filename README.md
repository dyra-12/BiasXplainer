# BiasXplainer (BiasGuard Pro) 🚀

AI-assisted text bias detection & explanation with SHAP visual insights, counterfactual rewrites, batch analytics, and comparative fairness exploration — all in one elegant Gradio interface.

<p align="center">
  <a href="https://huggingface.co/spaces/OWNER/BiasXplainer"><img alt="HF Space" src="https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-purple" /></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB" />
  <img alt="License" src="https://img.shields.io/badge/License-MIT-lightgray" />
  <img alt="Status" src="https://img.shields.io/badge/Status-Active-success" />
  <img alt="Model" src="https://img.shields.io/badge/Model-DistilBERT-blue" />
</p>

> Replace the Space URL badge if your namespace differs.

## ✨ Live Demo

[Launch on Hugging Face Spaces →](https://huggingface.co/spaces/OWNER/BiasXplainer) *(placeholder — supply real URL to finalize)*

## 🔍 Why BiasXplainer?

Traditional toxicity/bias classifiers give you a label — BiasXplainer shows you *why*, highlights contributing words, and proposes neutral rewrites automatically. Built for rapid experimentation and responsible language model evaluation.

## 🧩 Feature Highlights

| Area | Capability |
|------|------------|
| Single Text | Gauge score + SHAP impact bar + highlighted tokens |
| Explanations | SHAP token attribution with fallback heuristic keywords |
| Counterfactuals | Template + semantic replacements + FLAN‑T5 grammar polish |
| Batch Mode | Multi-text ingestion (.txt/.csv/.json) + summary stats |
| Background Jobs | Async processing with status polling (extensible to queues) |
| Comparative View | Simple group substring bias comparison |
| Exports | Download JSON/CSV via UI or programmatic helpers |
| Auto Model | Local `./models` override or DistilBERT fallback |
| Profiling | Per-step timing breakdown embedded in UI |

### Extended Highlights
- ⚡ Parallel bias + SHAP computation reduces latency.
- 🧠 Counterfactual rewrites neutralize gendered phrasing while preserving semantics.
- 📊 Aggregated batch summaries: mean bias, class counts, top impactful words.
- 🛠 Minimal, readable exporters (`export/*.py`) for downstream pipeline stitching.
- 🎨 Polished UX: custom CSS, semantic color states, accessible layout.

## 🏁 Quick Start (10s)

```bash
git clone https://github.com/dyra-12/BiasXplainer.git
cd BiasXplainer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Open the printed local URL and analyze text instantly.

## 🧪 Minimal API Usage (Programmatic)

```python
from main import BiasGuardDashboard

dashboard = BiasGuardDashboard()               # Loads models + explainer
texts = [
	 "Women should be nurses because they are compassionate.",
	 "Men are naturally better at engineering roles.",
	 "This is a neutral sentence."
]

results = dashboard.analyzer.detector.predict_batch_batched(texts)
print(results)  # [{'bias_probability': ..., 'classification': ..., 'confidence': ...}, ...]
```

## 🖥 How to Use (GUI)

1. Single Analysis
	- Paste text → Analyze → View gauge, SHAP chart, highlighted tokens, counterfactuals.
2. Batch & Compare
	- Provide multi-line text OR upload `.txt/.csv/.json` → start background job → poll → export.
3. Comparison
	- Supply two substrings (e.g. `women` vs `men`) to contrast average bias statistics.
4. Export
	- One-click JSON/CSV after a run or specify a path at job start.

### Supported Input Formats
- `.txt` newline-separated
- `.csv` with a `text` column (fallback to first column if absent)
- `.json` list of strings OR list of objects containing `text`

See `docs/usage.md` for advanced patterns (background jobs, performance tuning, fallbacks).

## 📦 Local Installation

Prerequisites
- Python 3.10+
- Optional GPU (CUDA) for faster SHAP + FLAN‑T5 inference

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## 🛠 Architecture Overview

```
┌─────────────────────────┐
│        User Input       │
└─────────────┬───────────┘
				  │ text
		┌───────▼────────┐
		│ BiasDetector    │ DistilBERT → bias score/class
		└───────┬────────┘
				  │ text (parallel)
		┌───────▼────────┐
		│ SHAPExplainer   │ SHAP token attributions / fallback keywords
		└───────┬────────┘
				  │ tokens + impacts
		┌───────▼────────┐
		│ Counterfactuals │ Template + replacements + FLAN‑T5 polish
		└───────┬────────┘
				  │ suggestions
		┌───────▼────────┐
		│  Dashboard UI  │ Plotly gauge, bar chart, highlights, profiling
		└────────────────┘
```

## 🔬 Technical Details

Core modules
- `core/bias_detector.py` — DistilBERT sequence classifier (batched + single inference)
- `core/explainer.py` — SHAP integration + resilient fallback keyword scorer
- `core/counterfactuals.py` — Suggestion engine + FLAN‑T5 grammar polishing
- `export/json_export.py`, `export/csv_export.py` — Simple serialization helpers
- `main.py` — Gradio composition (BiasGuardPro + BiasGuardDashboard)

Performance strategy
- Parallel futures for bias prediction + SHAP attribution
- Batched inference (`predict_batch_batched`) reduces tokenizer/model overhead
- Token merging to consolidate subwords into human-readable impact units
- Lightweight HTML profiling block surfaces top latency contributors

Models
- DistilBERT (local fine-tune if model artifacts in `./models`; fallback base)
- FLAN‑T5 Small (rewrite polishing)

## 📈 Performance

`results/latency_before_after.csv` exists as a placeholder for profiling exports (currently empty). You can populate it by logging before/after introducing batching & parallel futures.

Suggested metrics to record:
- Mean single-analysis latency (ms)
- SHAP attribution time (ms)
- Counterfactual generation time (ms)
- Batch throughput (texts/sec) at various batch sizes

## ✅ Testing

```bash
pytest -q
```

Add new tests under `tests/` for:
- Deterministic classification on neutral vs biased fixture texts
- SHAP fallback branch integrity when explainer fails
- Exporters producing expected headers

## 🧭 Roadmap

- Real-time websocket progress streaming
- Rich fairness metrics (e.g., equalized odds across groups)
- Persistent job queue (Celery / RQ + Redis) abstraction
- Embedding-based semantic grouping for comparisons
- Model calibration & confidence intervals
- Pluggable transformer backbone (RoBERTa / DeBERTa)

## 🤝 Contributing

1. Fork & branch: `feat/descriptive-name`
2. Keep patches atomic and well‑scoped
3. Add/adjust tests when changing logic
4. Submit PR with before/after performance notes if relevant

## ⚖️ Ethical Use & Disclaimer

This tool aids qualitative inspection of potential gender-related bias in text. It is **not** a definitive bias arbiter. Always combine automated signals with human review, domain context, and inclusive guidelines. Do not deploy unvalidated models for sensitive decision workflows.

## 🔐 Security Notes
- Avoid sending PII or confidential corpora to hosted public Spaces without review.
- For enterprise usage, run locally and restrict model artifact access.

## 📜 License

MIT (placeholder) — add `LICENSE` file for formal adoption.

---

If you share your actual Hugging Face Space URL, I’ll finalize the badge/link above.

## How to Use

1) Single analysis
- Enter text and click Analyze to get:
	- Bias score and classification
	- SHAP-based word impact chart + highlighted text
	- Counterfactual alternatives to make the sentence more neutral

2) Batch & Compare
- Paste multiple texts (one per line) or upload a file
- Optionally enter two group substrings to compare (e.g., "women" vs "men")
- Start a background job, then poll for status; download results after completion

3) Export
- Use Export JSON / Export CSV buttons after a run
- Files save to `./export/` (you can also specify a path when starting a job)

See `docs/usage.md` for a deeper walkthrough and troubleshooting tips.

## Local Installation

Prereqs
- Python 3.10+
- (Optional) GPU for faster SHAP/T5 operations

Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the app

```bash
python main.py
```

The Gradio UI opens in your browser. Use the Batch & Compare tab for multi-text jobs.

Supported inputs
- .txt: newline-separated texts
- .csv: include a `text` column (or the first text-like column is used)
- .json: list of strings or list of objects with a `text` field

## Technical Details

Core components
- `core/bias_detector.py` — DistilBERT classifier (Hugging Face Transformers)
	- Auto-detects a local model under `./models` and loads tokenizer/weights
	- Provides `predict_bias`, `predict_batch`, and `predict_batch_batched` for throughput
- `core/explainer.py` — SHAP-based token impact explainer
	- Uses `shap.maskers.Text` and a batched prediction wrapper for speed
	- Falls back to heuristic keywords when SHAP isn’t available
- `core/counterfactuals.py` — CounterfactualGenerator with a GrammarPolisher
	- Template-and-replacement suggestions guided by SHAP keywords
	- Polishes phrasing via FLAN‑T5‑Small and light grammar rules
- `export/json_export.py`, `export/csv_export.py` — simple exporters used by the UI
- `main.py` — Gradio UI (BiasGuardPro + BiasGuardDashboard)
	- Plotly meter, SHAP bar chart, highlighted text, profiling, and custom CSS

Data flow (single analysis)
1. `BiasDetector.predict_bias(text)` → bias probability/class
2. `SHAPExplainer.get_shap_values(text)` → per-token impact scores
3. `CounterfactualGenerator.generate_counterfactuals(text, shap)` → neutral alternatives
4. UI renders a gauge, bar chart, highlights, and suggestions

Batch flow
- Uses batched DistilBERT inference; supports textarea or file inputs
- Background jobs store results and summary (avg bias, class counts, top words)
- Optional simple substring-based comparative view

Models
- Classifier: DistilBERT (local from `./models` if present; else `distilbert-base-uncased`)
- Polisher: `google/flan-t5-small`

## Testing

```bash
pytest -q
```

## Roadmap

- Live streaming progress UI for long jobs
- Pluggable model backends and better calibration
- More robust comparison tooling and fairness metrics
- Optional persistence for background jobs (e.g., Redis/RQ or Celery)

---

If you share your actual Hugging Face Space URL, I’ll wire it into the badge/link above.
