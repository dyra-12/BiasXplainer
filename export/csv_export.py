"""export/csv_export.py

Simple CSV exporter used by batch processing and summary exports.
"""

import csv
from typing import Any, Dict, List


def export_results_to_csv(results: List[Dict[str, Any]]) -> str:
    """Return CSV content as a string for a list of result dicts.

    This keeps only a subset of fields for readability.
    """
    if not results:
        return ""

    # Choose columns (fallback to keys of first result)
    columns = ["text", "bias_probability", "bias_class", "confidence"]
    first = results[0]
    for col in columns:
        if col not in first:
            # Fall back to available keys
            columns = list(first.keys())[:4]
            break

    output_lines = []
    # write header
    output_lines.append(",".join(columns))

    for r in results:
        row = []
        for c in columns:
            val = r.get(c, "")
            if isinstance(val, str):
                v = val.replace('"', '""')
                if "," in v or "\n" in v:
                    v = f'"{v}"'
            else:
                v = str(val)
            row.append(v)
        output_lines.append(",".join(row))

    return "\n".join(output_lines)


def save_results_csv(path: str, results: List[Dict[str, Any]]) -> None:
    """Save results to a CSV file path."""
    csv_content = export_results_to_csv(results)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(csv_content)
