"""Lightweight PDF export helpers.

This module provides a minimal 'export to PDF' helper. To avoid adding a
heavy dependency (like reportlab) in this exercise the function will write
a simple plaintext representation to a .pdf file. For production use
replace this with a real PDF renderer.
"""

from typing import Any, Dict, List


def save_results_pdf(path: str, results: List[Dict[str, Any]]) -> None:
    """Write a plaintext representation of results to the given path.

    Note: This is a lightweight stub. The resulting file will be valid but
    not a true typeset PDF. Replace with a proper PDF generation flow if
    needed.

    Args:
            path: Output file path (recommended .pdf extension).
            results: List of result dictionaries to include.
    """
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"Text: {r.get('text', '')}\n")
            f.write(
                f"Bias: {r.get('bias_probability', '')} ({r.get('bias_class', '')})\n"
            )
            f.write("---\n")
