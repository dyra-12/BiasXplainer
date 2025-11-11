# BiasXplainer — Programmatic Interface Guide

This document describes the primary programmatic interfaces of BiasXplainer and how to use them from scripts or other Python projects.

## Main entry points

### BiasGuardPro

Class: `main.BiasGuardPro(model_path: Optional[str] = None)`

Primary convenience wrapper that wires together the detector, SHAP explainer and counterfactual generator.

Key methods:
- `analyze_text(text: str) -> Dict` — Run a full analysis of a single text. Returns a dict with keys described in `docs/usage.md`.
- `analyze_batch(texts: List[str], progress_callback: Optional[Callable] = None) -> List[Dict]` — Analyze multiple texts. `progress_callback(processed, total)` can be supplied to track progress.
- `summarize_batch(results: List[Dict]) -> Dict` — Aggregate statistics: total, avg bias, class counts and top words.
- `compare_groups(group_a: List[Dict], group_b: List[Dict]) -> Dict` — Simple group statistics and delta.

Exceptions:
- Functions attempt to catch internal failures and return partial results; however, fatal initialization errors (model loading) will raise exceptions at construction time.

### BiasGuardDashboard

Class: `main.BiasGuardDashboard()` — Builds the Gradio UI. Use `create_dashboard()` to get a `gradio.Blocks` instance and `launch()` to run the UI.

## Core components

### BiasDetector

Module: `core/bias_detector.py`

Class: `BiasDetector(model_path: str = '.')`

Purpose: Loads a transformer classifier and tokenizer for bias detection.

Important methods:
- `predict_bias(text: str) -> Dict` — Returns `{'bias_probability': float, 'classification': str, 'confidence': float}`.
- `predict_batch(texts: List[str]) -> List[Dict]` — Sequential per-item predictions.
- `predict_batch_batched(texts: List[str], batch_size: int = 8) -> List[Dict]` — Tokenizes and runs the model in batches for improved throughput.

Usage notes:
- The classifier threshold for `classification` uses 0.5 by default.
- The class moves the model to CUDA if available.

### SHAPExplainer

Module: `core/explainer.py`

Class: `SHAPExplainer(model_path: str = '.')`

Purpose: Provide per-word SHAP contributions for input text using the detector and SHAP's text masker.

Important methods:
- `get_shap_values(text: str, max_evals: int = 500) -> List[Tuple[str, float]]` — Returns a sorted list of (word, score) pairs. If SHAP initialization fails, a keyword-based fallback is used.

Notes:
- The class wraps a `model_predict` callable that returns a numpy array of shape (n,1) corresponding to bias probabilities. This makes it compatible with SHAP's Explainer API.
- `_combine_subword_scores` recombines BPE-style subword tokens into readable words.

### CounterfactualGenerator

Module: `core/counterfactuals.py`

Purpose: Generate neutral alternatives and small counterfactual rewrites to mitigate biased phrasing.

Important methods:
- `generate_counterfactuals(text: str, shap_results: List[Tuple[str, float]], num_alternatives: int = 3) -> List[str]` — Returns a list of alternative phrasings.

Customization:
- Edit `templates` and `replacements` in the class constructor to tune behavior for your domain.

## Export helpers

Modules in `export/`:
- `json_export.py` — `export_results_to_json(results) -> str`, `save_results_json(path, results) -> None`.
- `csv_export.py` — `export_results_to_csv(results) -> str`, `save_results_csv(path, results) -> None`.
- `clipboard_export.py` — `format_results_for_clipboard(results) -> str` and `copy_to_clipboard_text(results) -> str` (non-invasive; does not touch system clipboard).
- `pdf_export.py` — `save_results_pdf(path, results) -> None` (lightweight plaintext stub; replace with a real PDF library for production-grade output).

## Tips for integration

- Create a thin wrapper function in your application that calls `BiasGuardPro.analyze_text` and adapts the returned structure to your UI or downstream pipeline.
- For high-throughput pipelines use `predict_batch_batched` directly if you only need bias probabilities and do not require full SHAP explanations for every item.

## Error handling & logging

- Many functions attempt to gracefully fallback on errors (e.g., SHAP fallback). For strict behavior, wrap calls and raise or log errors as needed in your integration layer.

## Examples

Programmatic example combining detection and export:

```python
from main import BiasGuardPro
from export.json_export import save_results_json

analyzer = BiasGuardPro()
results = analyzer.analyze_batch(["...", "..."])
save_results_json("./exports/batch_results.json", results)
```

If you need any missing signatures or examples added here, tell me which functions you call most and I will expand this guide.
