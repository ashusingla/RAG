"""Chunk markdown text by heading sections.

A section starts at any line beginning with `# `, `## `, or `### ` and runs
until the next such line (or end of file). The chunk text includes the
heading line itself.
"""
from __future__ import annotations


_HEADING_PREFIXES = ("# ", "## ", "### ")


def chunk_by_section(text: str) -> list[dict]:
    """Split markdown text into per-section chunks.

    Returns a list of {"id": "section-N", "heading": str, "text": str}.
    Raises ValueError if the input has zero headings.
    """
    chunks: list[dict] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    def flush():
        if current_heading is not None:
            chunks.append({
                "id": f"section-{len(chunks)}",
                "heading": current_heading,
                "text": "\n".join(current_lines).strip(),
            })

    for line in text.splitlines():
        if any(line.startswith(p) for p in _HEADING_PREFIXES):
            flush()
            current_heading = line.lstrip("#").strip()
            current_lines = [line]
        elif current_heading is not None:
            current_lines.append(line)

    flush()

    if not chunks:
        raise ValueError(
            "No markdown heading found. Source must contain at least one "
            "line starting with '# ', '## ', or '### '."
        )
    return chunks
