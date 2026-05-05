import pytest
from chunker import chunk_by_section


def test_single_section():
    text = "# Only Section\nLine one.\nLine two.\n"
    chunks = chunk_by_section(text)
    assert len(chunks) == 1
    assert chunks[0]["id"] == "section-0"
    assert chunks[0]["heading"] == "Only Section"
    assert "Line one." in chunks[0]["text"]
    assert "Line two." in chunks[0]["text"]
    assert chunks[0]["text"].startswith("# Only Section")


def test_multiple_sections():
    text = (
        "# A\nbody a\n"
        "# B\nbody b1\nbody b2\n"
        "# C\nbody c\n"
    )
    chunks = chunk_by_section(text)
    assert [c["heading"] for c in chunks] == ["A", "B", "C"]
    assert [c["id"] for c in chunks] == ["section-0", "section-1", "section-2"]
    assert "body b1" in chunks[1]["text"]
    assert "body b2" in chunks[1]["text"]
    assert "body c" not in chunks[1]["text"]


def test_mixed_heading_levels():
    text = "# H1\nbody1\n## H2\nbody2\n### H3\nbody3\n"
    chunks = chunk_by_section(text)
    assert [c["heading"] for c in chunks] == ["H1", "H2", "H3"]


def test_zero_headings_raises():
    with pytest.raises(ValueError, match="heading"):
        chunk_by_section("just some text\nno headings here\n")


def test_text_before_first_heading_is_dropped():
    text = "preamble line\n# First\nbody\n"
    chunks = chunk_by_section(text)
    assert len(chunks) == 1
    assert "preamble" not in chunks[0]["text"]


def test_heading_with_no_body():
    text = "# Empty\n# Has Body\nx\n"
    chunks = chunk_by_section(text)
    assert chunks[0]["heading"] == "Empty"
    assert chunks[0]["text"].strip() == "# Empty"
    assert chunks[1]["heading"] == "Has Body"
