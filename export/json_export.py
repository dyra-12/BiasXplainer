"""
File: export/json_export.py
Currently empty - provide a simple exporter helper
"""
import json
from typing import List, Dict, Any


def export_results_to_json(results: List[Dict[str, Any]]) -> str:
	"""Return a pretty-printed JSON string of the results."""
	return json.dumps(results, indent=2, ensure_ascii=False)


def save_results_json(path: str, results: List[Dict[str, Any]]) -> None:
	"""Save results to a file path as JSON."""
	with open(path, 'w', encoding='utf-8') as f:
		json.dump(results, f, indent=2, ensure_ascii=False)
