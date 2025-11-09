<div align="center">
  
<img src="assets/biasxplainer_banner.png" alt="BiasXplainer Dashboard Preview" width="820"/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg)
![Transformers](https://img.shields.io/badge/🤖-Transformers-yellow.svg)
![Gradio](https://img.shields.io/badge/Gradio-4.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)
![Model](https://img.shields.io/badge/Base%20Model-DistilBERT-blue.svg)

**An Explainable Text Bias Auditing & Counterfactual Suggestion Toolkit**

[🚀 Live Demo](#live-demo) | [✨ Features](#features) | [🎯 Usage Guide](#usage-guide) | [🧠 Technical Details](#technical-details) | [🤝 Contributing](#contributing)

</div>

---

## 🌟 Overview

**BiasXplainer** (a.k.a. BiasGuard Pro) is an interactive and programmatic toolkit for:
- 🔍 Detecting potential bias (e.g. gendered language) in short text inputs
- 🧪 Explaining classifier outputs via token-level SHAP attributions (with graceful fallbacks)
- 🔁 Generating neutral counterfactual rewrites while preserving semantics
- 📊 Running batch analyses with aggregation statistics & comparative subgroup views
- 🛠 Exporting structured outputs (JSON/CSV) for downstream pipelines

This toolkit is designed as an **exploratory auditing aid**—ideal for rapid experimentation, prototype fairness evaluation, educational demonstrations, and workflow integration proof-of-concepts.

> ⚠️ **Important**: This is **not** a definitive bias measurement instrument. For production or compliance use:
> - Apply validated, domain-appropriate bias and fairness metrics
> - Use diverse, representative corpora (not isolated sentences)
> - Incorporate human review and domain expertise
> - Follow established AI ethics & governance frameworks

---

## 🎯 Purpose & Scope

BiasXplainer helps you:
- Understand *why* a sentence may be flagged: token impacts clearly visualized
- Explore neutral alternatives via structured counterfactual suggestions
- Analyze variability across groups in batch mode (substring-based comparative lens)
- Export artifacts for integration with other auditing pipelines

---

## 🎭 Why This Toolkit?

| Goal | How It's Achieved |
|------|-------------------|
| Transparency | Token-level impact via SHAP + heuristic fallback |
| Counterfactual Exploration | Template + semantic replacements + FLAN‑T5 polish |
| Batch Insight | Aggregated stats: mean bias, class distribution, top impactful terms |
| Fairness Prototyping | Simple comparative view (e.g., “women” vs “men” substring focus) |
| Extensibility | Modular architecture for adding models / exporters / fairness metrics |
| Developer Friendliness | Clean Python modules + minimal dependency surface |

---

<a id="features"></a>
## ✨ Features

### 🔬 Core Capabilities
- **Single Text Analysis**: Bias score + classification + SHAP impact chart + highlighted tokens
- **Counterfactual Suggestions**: Structured rewrite candidates (neutralization focus)
- **Batch Mode**: Accepts `.txt`, `.csv`, `.json` and runs background jobs
- **Comparative View**: Substring-based group comparison (basic fairness proxy)
- **Exports**: JSON & CSV from UI or programmatic API
- **Model Override**: Auto-load local fine-tuned DistilBERT if present under `./models/`
- **Profiling Panel**: Inline latency breakdown (tokenization, SHAP, rewrite phases)

### 🧩 Extended Features
- ⚡ Parallel bias + SHAP computation
- 🧪 Resilient fallback keyword scoring if SHAP fails
- 🔁 FLAN‑T5-backed grammar polishing for counterfactuals
- 🧱 Batched inference APIs (`predict_batch_batched`)
- 🧪 Hooks for future fairness metrics (equalized odds, subgroup performance gaps)

---

<a id="live-demo"></a>
## 🚀 Live Demo

Try it instantly on Hugging Face Spaces:

### 👉 [Launch Interactive Demo](https://huggingface.co/spaces/OWNER/BiasXplainer)

*(Replace `OWNER` with your actual namespace. Provide the real URL and this section will be finalized.)*

---

## 🗂 Project Structure

```
BiasXplainer/
│
├── main.py                      # Gradio entrypoint (BiasGuardDashboard)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── core/
│   ├── bias_detector.py         # DistilBERT classifier logic
│   ├── explainer.py             # SHAP integration + fallback heuristic
│   ├── counterfactuals.py       # Rewrite engine + FLAN‑T5 polishing
│   ├── utils.py                 # Helpers (token merging, formatting, etc.)
│
├── export/
│   ├── json_export.py           # JSON serialization helper
│   ├── csv_export.py            # CSV serialization helper
│
├── models/                      # (Optional) Local fine-tuned model artifacts
├── tests/                       # Pytest suite (add fairness & stability tests)
├── docs/
│   ├── usage.md                 # Advanced usage patterns
│   ├── api.md                   # Programmatic interface reference
│   └── roadmap.md               # Extended roadmap details
│
├── assets/                      # (Optional) Screenshots / banners
├── results/
│   └── latency_before_after.csv # Placeholder performance metrics file
└── LICENSE                      # MIT license (add full text)
```

---

<a id="usage-guide"></a>
## 🎯 Usage Guide

### Quick Start (10 Seconds)
```bash
git clone https://github.com/dyra-12/BiasXplainer.git
cd BiasXplainer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```
Open the printed local URL to start auditing text.

### GUI Workflows

1. **Single Analysis**
   - Paste text → Analyze → View bias score, class, SHAP token chart, highlighted impacts, and counterfactual suggestions.

2. **Batch & Compare**
   - Paste multi-line text OR upload `.txt / .csv / .json` → Start job → Poll status → Download results.
   - Optionally specify two substrings (e.g., `women`, `men`) for comparative averages.

3. **Export**
   - Use Export buttons (JSON/CSV) or programmatic API calls.
   - Files saved under `./export/` by default.

### Supported Input Formats
| Format | Structure |
|--------|-----------|
| `.txt` | Newline-separated sentences |
| `.csv` | Must include a `text` column (fallback to first column) |
| `.json` | List of strings OR list of objects with `text` field |

### Minimal Programmatic API
```python
from main import BiasGuardDashboard

dashboard = BiasGuardDashboard()
texts = [
    "Women should be nurses because they are compassionate.",
    "Men are naturally better at engineering roles.",
    "This is a neutral sentence."
]

results = dashboard.analyzer.detector.predict_batch_batched(texts)
print(results)  # [{'bias_probability': ..., 'classification': ..., 'confidence': ...}, ...]
```

---

## 🧪 Example Analysis Flow

```
1. User enters text
2. BiasDetector.predict_bias(text) → {probability, class, confidence}
3. Explainer.get_shap_values(text) → token-level impacts
4. CounterfactualGenerator.generate_counterfactuals(text, shap) → neutral rewrite candidates
5. UI renders gauge + impact bar + highlighted text + suggestions + profiling
```

### Batch Flow
- Batched DistilBERT inference
- Async background job collects:
  - Mean bias probability
  - Class distribution counts
  - Top impactful tokens
- Optional simple substring comparison overlay

---

<a id="technical-details"></a>
## 🧠 Technical Details

### Core Modules
| Module | Responsibility |
|--------|----------------|
| `core/bias_detector.py` | DistilBERT-based classification (single + batched + efficient batching) |
| `core/explainer.py` | SHAP token attribution + fallback keyword heuristic |
| `core/counterfactuals.py` | Template-driven neutralization + FLAN‑T5 grammar refinement |
| `export/json_export.py` | Structured JSON serialization |
| `export/csv_export.py` | CSV export with impact flattening |
| `main.py` | Gradio composition, UI orchestration, profiling |

### Models
- **Classifier**: DistilBERT (`distilbert-base-uncased` or local fine-tune under `./models`)
- **Polisher**: `google/flan-t5-small` (light rewrite improvements)

### Performance Strategy
- Parallel futures for classifier + SHAP tasks
- Batching reduces tokenizer & forward overhead
- Token merging converts subword fragments into user-friendly units
- Inline profiling block surfaces latency bottlenecks

### Counterfactual Strategy
1. Extract high-impact tokens (SHAP or fallback keyword list)
2. Apply neutral replacements or paraphrase templates
3. Polish grammar & semantics via FLAN‑T5 Small
4. Return ranked suggestions (preserving original intent)

---

## 📊 Performance

Record metrics in `results/latency_before_after.csv` (create if missing):

Suggested columns:
| metric | unit | description |
|--------|------|-------------|
| single_inference_mean | ms | Avg latency per sentence (no batching) |
| shap_explanation_mean | ms | Avg token attribution time |
| counterfactual_gen_mean | ms | Avg time to produce rewrites |
| batch_throughput_bsz_16 | texts/sec | Throughput at batch size 16 |
| batch_throughput_bsz_64 | texts/sec | Throughput at batch size 64 |

Add before/after rows when optimizing.

---

## ✅ Testing

```bash
pytest -q
```

Recommended test categories:
- Neutral vs biased fixtures (classification stability)
- SHAP fallback path when primary explainer errors
- Export correctness (headers + row counts)
- Deterministic counterfactual generation for controlled inputs

---

## 🗺 Roadmap

Short-Term:
- Live progress streaming (websocket / SSE)
- Extended fairness metrics (e.g., equalized odds, subgroup delta charts)
- Persistent job queue (Celery / RQ + Redis)
- Confidence calibration & uncertainty indicators
- Pluggable backbone registry (RoBERTa, DeBERTa, ALBERT)

Long-Term:
- Embedding-based semantic group comparison
- Multi-lingual model support
- Explanation fusion (SHAP + Integrated Gradients comparison)
- Audit session export bundles (results + metadata + configuration hash)

---

## 🎓 Use Cases

### Research
- Rapid prototyping of bias detection workflows
- Comparing attribution stability across variants

### Industry
- Early-stage content moderation tool exploration
- Internal fairness experimentation sandbox

### Education
- Teaching interpretability concepts interactively
- Student projects on ethical AI + XAI

---

<a id="contributing"></a>
## 🤝 Contributing

We welcome improvements!

### Ways to Contribute
1. 🐛 Bug Reports: Open an issue with reproduction steps
2. ✨ Feature Requests: Suggest metrics, exporters, or model options
3. 📝 Documentation: Improve guides, add examples
4. 💻 Code: Submit well-scoped PRs with tests
5. 📊 Performance: Optimize latency & update benchmarking CSV

### Development Setup
```bash
# Fork the repo, then:
git clone https://github.com/YOUR-USERNAME/BiasXplainer.git
cd BiasXplainer

# Create a branch
git checkout -b feat/your-feature-name

# Install & test
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q

# Commit & push
git commit -m "Feat: descriptive summary"
git push origin feat/your-feature-name
```

### Code Style
- Follow PEP 8
- Use type hints where practical
- Add docstrings to public functions
- Include tests for new logic paths
- Avoid large, multi-purpose PRs

---

## ⚖️ Ethical Use & Disclaimer

This toolkit provides **heuristic insights** into potential linguistic bias patterns. It does **not** guarantee:
- Fairness across real-world demographic groups
- Complete coverage of subtle stereotypes
- Context-aware ethical judgments

Always complement automated signals with:
- Human expert review
- Diverse, representative evaluation sets
- Formal fairness metrics and governance policies

---

## 🔒 Security Notes

- Avoid submitting PII or confidential corpora to public hosted demos.
- For enterprise usage: run locally, restrict model paths, audit dependencies.
- Consider dependency pinning & vulnerability scanning (e.g., `pip-audit`, `safety`).

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file.

```
MIT License

Copyright (c) 2024 BiasXplainer Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
[Full license text here]
```

---

## 🛟 Additional Resources

- `docs/usage.md` — Advanced usage & troubleshooting
- `docs/api.md` — Programmatic interface guide
- `docs/roadmap.md` — Expanded roadmap details (optional)
- Hugging Face Transformers Docs: https://huggingface.co/docs/transformers
- SHAP Documentation: https://shap.readthedocs.io/

---

## 📬 Contact & Support

- GitHub Issues: (Add URL once repo issues enabled)
- Discussions: (Enable discussions for Q&A if desired)
- Email: your_email@example.com (replace)
- Hugging Face Space: (Provide final live link)

---

<div align="center">

**Built with ❤️ by the community**

[⬆ Back to Top](#-overview)

</div>