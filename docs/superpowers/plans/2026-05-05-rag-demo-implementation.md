# RAG Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable Jupyter notebook in `~/Desktop/rag-demo/` that demonstrates the five canonical RAG steps end-to-end using Voyage AI embeddings and ChromaDB.

**Architecture:** A small Python module `chunker.py` (unit-tested) handles markdown-section chunking. A build script generates `rag_demo.ipynb` deterministically from cell sources, so the notebook is reproducible from version control. The notebook itself imports `chunker.py`, calls Voyage AI for embeddings, and stores them in a persistent ChromaDB collection on disk. Persistence is verified by a kernel-restart test.

**Tech Stack:** Python 3.10+, `voyageai` (embeddings), `chromadb` (vector store), `nbformat` (notebook generation), `python-dotenv` (env loading), `pytest` (tests), `jupyter` (notebook runtime).

---

## File Structure

| Path | Purpose | Owner |
|---|---|---|
| `requirements.txt` | Pinned deps for the demo | Task 1 |
| `.gitignore` | Excludes `.env`, `chroma_db/`, `.venv/`, `__pycache__/`, `.ipynb_checkpoints/` | Task 1 |
| `.env.example` | Template for `VOYAGE_API_KEY` | Task 1 |
| `demo.md` | Source text — 4 markdown sections on disjoint topics | Task 2 |
| `chunker.py` | `chunk_by_section(text)` pure function | Task 3 |
| `tests/test_chunker.py` | Pytest tests for chunker | Task 3 |
| `build_notebook.py` | Script that constructs `rag_demo.ipynb` from cell sources via `nbformat` | Task 4 |
| `rag_demo.ipynb` | Generated notebook (executed end-to-end in Task 6) | Task 4 (generated), Task 6 (executed) |

The chunker lives in its own `.py` file (not inline in the notebook) so it can be unit-tested with pytest. The notebook imports it. This is the same separation Quark's `ai-services-backend` uses for its semantic-search components.

---

## Task 1: Project scaffold (deps, gitignore, env template)

**Files:**
- Create: `~/Desktop/rag-demo/requirements.txt`
- Create: `~/Desktop/rag-demo/.gitignore`
- Create: `~/Desktop/rag-demo/.env.example`

- [ ] **Step 1: Write `requirements.txt`**

```
voyageai>=0.2.0
chromadb>=0.5.0
python-dotenv>=1.0.0
jupyter>=1.0.0
nbformat>=5.9.0
pytest>=8.0.0
```

- [ ] **Step 2: Write `.gitignore`**

```
.env
.venv/
chroma_db/
__pycache__/
.ipynb_checkpoints/
*.pyc
.pytest_cache/
```

- [ ] **Step 3: Write `.env.example`**

```
VOYAGE_API_KEY=your_key_here
```

- [ ] **Step 4: Commit**

```bash
cd ~/Desktop/rag-demo
git add requirements.txt .gitignore .env.example
git commit -m "chore: project scaffold (deps, gitignore, env template)"
```

---

## Task 2: Demo source content

**Files:**
- Create: `~/Desktop/rag-demo/demo.md`

- [ ] **Step 1: Write `demo.md`**

```markdown
# Photosynthesis

Plants convert sunlight into chemical energy through photosynthesis. Chlorophyll in the leaves absorbs light, and the plant uses that energy to combine carbon dioxide from the air with water from the soil to produce glucose and oxygen. Glucose feeds the plant; oxygen is released into the atmosphere as a byproduct.

# The Roman Aqueducts

Ancient Rome built a vast network of aqueducts to supply fresh water to its cities. The aqueducts relied entirely on gravity, channeling water from distant springs and rivers along precisely graded stone channels. At their peak, they delivered hundreds of millions of liters of water per day to public baths, fountains, and private homes.

# Bitcoin Mining

Bitcoin mining secures the network through proof-of-work. Miners compete to find a hash value below a target threshold by repeatedly varying a nonce in the candidate block header. The first miner to find a valid hash broadcasts the new block to the network and earns the block reward plus transaction fees.

# Honeybee Communication

Honeybees use a "waggle dance" to communicate the direction and distance of food sources to other workers. The angle of the dance relative to vertical encodes the direction relative to the sun, and the duration of the waggle phase encodes the distance. Other bees follow the dance and then fly out to find the food.
```

- [ ] **Step 2: Commit**

```bash
git add demo.md
git commit -m "feat: add demo.md with four disjoint sections"
```

---

## Task 3: Chunker module via TDD

**Files:**
- Create: `~/Desktop/rag-demo/tests/__init__.py` (empty file)
- Create: `~/Desktop/rag-demo/tests/test_chunker.py`
- Create: `~/Desktop/rag-demo/chunker.py`

The chunker walks the markdown line by line. A line that starts with `# `, `## `, or `### ` opens a new section. Each section's `text` is the heading line plus all body lines until the next heading. Any text appearing *before* the first heading is dropped (we explicitly test this so the behavior is intentional, not accidental). If the input has zero headings, raise `ValueError`.

- [ ] **Step 1: Set up venv and install pytest**

```bash
cd ~/Desktop/rag-demo
python -m venv .venv
source .venv/bin/activate
pip install pytest
```

- [ ] **Step 2: Write the failing tests**

Create `tests/__init__.py` as an empty file.

Create `tests/test_chunker.py`:

```python
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
```

- [ ] **Step 3: Run tests — verify they fail**

```bash
pytest tests/test_chunker.py -v
```

Expected: All 6 tests fail with `ModuleNotFoundError: No module named 'chunker'`.

- [ ] **Step 4: Implement `chunker.py`**

```python
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
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
pytest tests/test_chunker.py -v
```

Expected: All 6 tests pass.

- [ ] **Step 6: Commit**

```bash
git add chunker.py tests/__init__.py tests/test_chunker.py
git commit -m "feat: chunker.chunk_by_section with markdown-heading splits"
```

---

## Task 4: Notebook build script and generated notebook

**Files:**
- Create: `~/Desktop/rag-demo/build_notebook.py`
- Create (via running the script): `~/Desktop/rag-demo/rag_demo.ipynb`

`build_notebook.py` defines each cell as a Python string and uses `nbformat` to assemble a clean `rag_demo.ipynb`. This makes the notebook regeneratable from source — if we want to tweak a cell, edit the script and re-run it.

- [ ] **Step 1: Install notebook deps**

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Expected: `voyageai`, `chromadb`, `python-dotenv`, `jupyter`, `nbformat`, `pytest` installed without errors.

- [ ] **Step 2: Write `build_notebook.py`**

```python
"""Generate rag_demo.ipynb from cell sources defined here.

Run: python build_notebook.py
Result: rag_demo.ipynb ready to open in Jupyter.
"""
from __future__ import annotations
from pathlib import Path
import nbformat as nbf


INTRO_MD = """\
# RAG Demo

Five canonical RAG steps end-to-end with Voyage AI embeddings and ChromaDB.

## Setup (run once before opening this notebook)

```bash
cd ~/Desktop/rag-demo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then edit .env and put your real Voyage key in
jupyter notebook rag_demo.ipynb
```

Then run cells top-to-bottom.

To simulate the *"some time later"* gap from step 4, restart the kernel after
cell 5 and run only cells 1, 6, 7. Chroma reopens the on-disk store and
retrieval still works.
"""

CELL_1_SETUP = '''\
# Cell 1 — Setup
import os
from pathlib import Path
from dotenv import load_dotenv
import voyageai
import chromadb
from chunker import chunk_by_section

load_dotenv()
if not os.getenv("VOYAGE_API_KEY"):
    raise RuntimeError(
        "VOYAGE_API_KEY missing. Copy .env.example to .env and set the key."
    )

PROJECT_DIR = Path.cwd()
DEMO_FILE = PROJECT_DIR / "demo.md"
CHROMA_DIR = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "demo"
EMBED_MODEL = "voyage-3-lite"

print("Setup OK. Voyage key loaded; paths configured.")
print(f"  PROJECT_DIR = {PROJECT_DIR}")
print(f"  CHROMA_DIR  = {CHROMA_DIR}")
'''

CELL_2_LOAD = '''\
# Cell 2 — Load demo.md
text = DEMO_FILE.read_text()
print(f"Loaded {len(text)} chars from {DEMO_FILE.name}\\n")
print("--- preview (first 300 chars) ---")
print(text[:300])
'''

CELL_3_CHUNK = '''\
# Cell 3 — Step 1: chunk by section
chunks = chunk_by_section(text)
print(f"Got {len(chunks)} chunks:\\n")
for c in chunks:
    print(f"  [{c['id']}] {c['heading']!r} ({len(c['text'])} chars)")
'''

CELL_4_EMBED = '''\
# Cell 4 — Step 2: embed each chunk via Voyage AI
client = voyageai.Client()  # picks up VOYAGE_API_KEY from env
result = client.embed(
    texts=[c["text"] for c in chunks],
    model=EMBED_MODEL,
    input_type="document",
)
embeddings = result.embeddings
print(f"Embedded {len(embeddings)} chunks.")
print(f"Each embedding has {len(embeddings[0])} dimensions.")
print(f"Total tokens used: {result.total_tokens}")
'''

CELL_5_STORE = '''\
# Cell 5 — Step 3: create vector DB and add embeddings
chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))

# Uncomment to wipe and re-index from scratch (e.g., after editing demo.md):
# chroma.delete_collection(COLLECTION_NAME)

collection = chroma.get_or_create_collection(name=COLLECTION_NAME)
collection.add(
    ids=[c["id"] for c in chunks],
    embeddings=embeddings,
    documents=[c["text"] for c in chunks],
    metadatas=[{"heading": c["heading"]} for c in chunks],
)
print(f"Collection {COLLECTION_NAME!r} now contains {collection.count()} items.")
print(f"Persisted to: {CHROMA_DIR}")
'''

CELL_6_QUERY_EMBED = '''\
# Cell 6 — Step 4: a user asks a question; embed it
question = "How do plants make food?"

q_emb = client.embed(
    texts=[question],
    model=EMBED_MODEL,
    input_type="query",
).embeddings[0]

print(f"Question: {question}")
print(f"Query embedding has {len(q_emb)} dimensions.")
'''

CELL_7_RETRIEVE = '''\
# Cell 7 — Step 5: retrieve the 2 most relevant chunks
results = collection.query(query_embeddings=[q_emb], n_results=2)

print(f"Top 2 results for: {question!r}\\n")
for i in range(len(results["ids"][0])):
    chunk_id = results["ids"][0][i]
    distance = results["distances"][0][i]
    heading = results["metadatas"][0][i]["heading"]
    text_preview = results["documents"][0][i][:200].replace("\\n", " ")
    print(f"#{i+1} [{chunk_id}] {heading} (distance={distance:.4f})")
    print(f"   {text_preview}...\\n")
'''


def build():
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(INTRO_MD),
        nbf.v4.new_code_cell(CELL_1_SETUP),
        nbf.v4.new_code_cell(CELL_2_LOAD),
        nbf.v4.new_code_cell(CELL_3_CHUNK),
        nbf.v4.new_code_cell(CELL_4_EMBED),
        nbf.v4.new_code_cell(CELL_5_STORE),
        nbf.v4.new_code_cell(CELL_6_QUERY_EMBED),
        nbf.v4.new_code_cell(CELL_7_RETRIEVE),
    ]
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    out = Path(__file__).parent / "rag_demo.ipynb"
    nbf.write(nb, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    build()
```

- [ ] **Step 3: Run the build script**

```bash
python build_notebook.py
```

Expected output: `Wrote /home/it/Desktop/rag-demo/rag_demo.ipynb`

- [ ] **Step 4: Verify the notebook is valid**

```bash
python -c "import nbformat; nb = nbformat.read('rag_demo.ipynb', as_version=4); print(f'cells={len(nb.cells)}'); nbformat.validate(nb); print('valid')"
```

Expected: `cells=8` (intro markdown + 7 code cells), `valid`.

- [ ] **Step 5: Commit**

```bash
git add build_notebook.py rag_demo.ipynb
git commit -m "feat: build_notebook.py and generated rag_demo.ipynb"
```

---

## Task 5: Configure environment for live run

**Files:**
- Create: `~/Desktop/rag-demo/.env` (NOT committed — listed in `.gitignore`)

- [ ] **Step 1: Get a Voyage API key**

If the user doesn't already have one:
1. Visit https://www.voyageai.com/ and sign up (free tier is sufficient).
2. Generate an API key in the dashboard.

If running this plan as an agent: stop here and ask the user for the key. Do not proceed to step 2 until you have it.

- [ ] **Step 2: Write `.env`**

```bash
cd ~/Desktop/rag-demo
cp .env.example .env
# Then edit .env and replace `your_key_here` with the real key.
```

- [ ] **Step 3: Verify the key is loaded (no commit)**

```bash
source .venv/bin/activate
python -c "from dotenv import load_dotenv; import os; load_dotenv(); k = os.getenv('VOYAGE_API_KEY'); print('OK' if k and k != 'your_key_here' else 'MISSING/PLACEHOLDER')"
```

Expected: `OK`

`.env` is git-ignored — no commit.

---

## Task 6: Execute the notebook end-to-end

**Files:**
- Modify (executed in place): `~/Desktop/rag-demo/rag_demo.ipynb`

This task runs the notebook from cell 1 through cell 7 in order, verifies each output, and commits the executed notebook (with outputs embedded).

- [ ] **Step 1: Launch Jupyter**

```bash
cd ~/Desktop/rag-demo
source .venv/bin/activate
jupyter notebook rag_demo.ipynb
```

The browser opens the notebook.

- [ ] **Step 2: Run cell 1 (Setup) and verify**

Click cell 1, press Shift+Enter.

Expected output:
```
Setup OK. Voyage key loaded; paths configured.
  PROJECT_DIR = /home/it/Desktop/rag-demo
  CHROMA_DIR  = /home/it/Desktop/rag-demo/chroma_db
```

If it raises `RuntimeError: VOYAGE_API_KEY missing`, fix `.env` (Task 5) and re-run.

- [ ] **Step 3: Run cell 2 (Load) and verify**

Expected: prints `Loaded NNNN chars from demo.md` followed by the first 300 chars of `demo.md` (starting with `# Photosynthesis`).

- [ ] **Step 4: Run cell 3 (Step 1 — chunk) and verify**

Expected output:
```
Got 4 chunks:

  [section-0] 'Photosynthesis' (NNN chars)
  [section-1] 'The Roman Aqueducts' (NNN chars)
  [section-2] 'Bitcoin Mining' (NNN chars)
  [section-3] 'Honeybee Communication' (NNN chars)
```

If chunk count is not 4 or headings are wrong, debug `chunker.py` against the real `demo.md`.

- [ ] **Step 5: Run cell 4 (Step 2 — embed) and verify**

Expected:
```
Embedded 4 chunks.
Each embedding has 512 dimensions.
Total tokens used: ~150-300
```

(Voyage `voyage-3-lite` returns 512-dim vectors. Token count varies with content length.) On a typical home connection this cell takes ~1-2 seconds.

If it raises `voyageai.error.AuthenticationError`, the API key in `.env` is wrong.

- [ ] **Step 6: Run cell 5 (Step 3 — store) and verify**

Expected:
```
Collection 'demo' now contains 4 items.
Persisted to: /home/it/Desktop/rag-demo/chroma_db
```

After this cell, verify the directory exists:

```bash
ls ~/Desktop/rag-demo/chroma_db/
```

Expected: a `chroma.sqlite3` file plus a UUID-named directory.

- [ ] **Step 7: Run cell 6 (Step 4 — query embed) and verify**

Expected:
```
Question: How do plants make food?
Query embedding has 512 dimensions.
```

- [ ] **Step 8: Run cell 7 (Step 5 — retrieve) and verify**

Expected first line: `Top 2 results for: 'How do plants make food?'`

Result #1 MUST be `[section-0] Photosynthesis` (it's the topical match). Result #2 will be one of the other three; the exact runner-up isn't important as long as #1 is Photosynthesis.

If Photosynthesis isn't ranked #1, something is wrong — most likely the `input_type` was set to `document` instead of `query` in cell 6. Fix and re-run cells 6 and 7.

- [ ] **Step 9: Save the notebook with outputs**

In Jupyter: File → Save Notebook (or Ctrl+S). The cell outputs are now embedded in `rag_demo.ipynb`.

- [ ] **Step 10: Commit the executed notebook**

```bash
cd ~/Desktop/rag-demo
git add rag_demo.ipynb
git commit -m "chore: commit executed notebook with cell outputs"
```

---

## Task 7: Persistence test ("some time later")

This validates that step 4's "some time later" gap is real — the on-disk Chroma store survives a kernel restart.

- [ ] **Step 1: Restart the Jupyter kernel**

In Jupyter: Kernel → Restart. (Do NOT clear outputs.)

- [ ] **Step 2: Run cell 1 only (Setup)**

Expected: same Setup OK output as before.

- [ ] **Step 3: Add a "reopen collection" helper cell**

Cell 6 alone would run fine — it only needs `client` and `EMBED_MODEL`, both defined in cell 1. But cell 7 references `collection`, which was defined in cell 5 and is gone after the kernel restart. To prove persistence without re-indexing, we need to reopen the existing on-disk collection.

In Jupyter, insert a new code cell between cells 5 and 6 (Insert → Insert Cell Below from cell 5) and paste:

```python
# Persistence-test helper: reopen the existing collection from disk
chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma.get_or_create_collection(name=COLLECTION_NAME)
print(f"Reopened collection {COLLECTION_NAME!r} with {collection.count()} items.")
```

Run it. Expected: `Reopened collection 'demo' with 4 items.` This proves the embeddings persisted.

- [ ] **Step 4: Run cell 6 (Step 4) and cell 7 (Step 5)**

Expected: same top-2 retrieval result as in Task 6, with Photosynthesis ranked #1. No re-indexing was needed.

- [ ] **Step 5: Optional — try a different question**

In cell 6, change `question = "How do plants make food?"` to one of:
- `"What did ancient Romans build to move water?"` → expect `The Roman Aqueducts` ranked #1
- `"How do bees tell each other where flowers are?"` → expect `Honeybee Communication` ranked #1
- `"What stops Bitcoin transactions from being faked?"` → expect `Bitcoin Mining` ranked #1

Re-run cells 6 and 7. Each query should pick the topically-correct chunk. If the correct chunk is in the top 2 for all four queries, retrieval is working.

- [ ] **Step 6: Save and commit (if you added the persistence helper cell)**

```bash
git add rag_demo.ipynb
git commit -m "feat: add persistence-test helper cell"
```

If you did NOT modify the notebook, no commit is needed.

---

## Done

After Task 7, `~/Desktop/rag-demo/` contains:
- A working notebook implementing all 5 RAG steps
- A unit-tested chunker module
- A persistent Chroma store proving the index survives restarts
- A reproducible build script for the notebook
- Six git commits telling the story end-to-end
