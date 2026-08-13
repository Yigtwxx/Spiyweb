"""Prompt templates, kept apart from pipeline logic.

Templates are plain `str.format` strings. They ask for machine-parseable plain
text (one item per line), never JSON - parsing stays trivial and `eval`-free.
"""

from __future__ import annotations

ENTITY_EXTRACTION_PROMPT = """\
Extract the named entities (people, organizations, places, products, events,
works, laws) mentioned in the following text.

Rules:
- Output ONE entity per line.
- Output ONLY the entity names - no commentary, no numbering, no explanations.
- If the text mentions no entities, output nothing.

Text:
{text}
"""
