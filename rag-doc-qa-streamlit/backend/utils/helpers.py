"""
backend/utils/helpers.py
Small utility functions used across the project.
"""

from pathlib import Path
from typing import Optional


def format_source_label(source: str, page: Optional[int] = None) -> str:
    """Return a human-readable source label."""
    label = Path(source).name
    if page is not None:
        label += f" (page {page})"
    return label


def truncate(text: str, max_chars: int = 300) -> str:
    """Truncate text to max_chars, appending ellipsis if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def confidence_emoji(level: str) -> str:
    """Return an emoji for a confidence level string."""
    return {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(level.upper(), "⚪")
