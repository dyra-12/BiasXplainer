"""Helpers for preparing and (optionally) copying results to the clipboard.

This module intentionally avoids external clipboard dependencies to keep the
package lightweight; it returns a formatted string that callers can copy
using their preferred mechanism.
"""

from typing import Any, Dict, List


def format_results_for_clipboard(results: List[Dict[str, Any]]) -> str:
    """Return a human-readable string representation of `results` suitable
    for placing on the clipboard.

    Args:
            results: List of result dictionaries to format.

    Returns:
            A single string with a readable summary of each result.
    """
    lines = []
    for r in results:
        text = r.get("text", "")
        prob = r.get("bias_probability", "")
        cls = r.get("bias_class", "")
        lines.append(f"Text: {text}\nBias: {prob} ({cls})\n---")
    return "\n".join(lines)


def copy_to_clipboard_text(results: List[Dict[str, Any]]) -> str:
    """Format results and return the string to be copied by the caller.

    This function deliberately does NOT interact with the system clipboard so
    it remains safe to call in headless or CI environments.
    """
    return format_results_for_clipboard(results)
