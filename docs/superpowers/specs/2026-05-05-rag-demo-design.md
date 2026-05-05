# RAG Demo — Design

**Date:** 2026-05-05
**Author:** asingla@quark.com
**Status:** Draft (awaiting user approval)

## Goal

Build a small, runnable Jupyter notebook that demonstrates the five canonical
steps of a retrieval-augmented-generation (RAG) pipeline end-to-end, using
only free models and libraries, on the user's local machine.

The five steps:

1. Chunk the source text by section.
2. Generate an embedding for each chunk.
3. Create a vector database and add each embedding to it.
4. Some time later, a user asks a question and an embedding is generated for it.
5. Search the vector store with that embedding and return the two most relevant chunks.

## Non-goals

- No LLM generation step. The "G" in RAG is intentionally out of scope; this
  is a retrieval demo only.
- No production-grade chunking (no overlap, no token-aware splitting, no
  semantic chunking).
- No reranking, hybrid search, metadata filtering, or evaluation harness.
- No web UI, no API server.

## Stack

| Concern | Choice | Why |
|---|---|---|
| Embeddings | Voyage AI (`voyage-3-lite`, 512 dims) | Free tier (~200M tokens/month), high-quality embeddings, simple Python SDK |
| Vector store | ChromaDB (`PersistentClient`) | File-based persistence out of the box, zero infrastructure, makes step 4's "some time later" real |
| Notebook | Jupyter | One cell per logical step; cells re-runnable independently |
| Env | `python-dotenv` for `VOYAGE_API_KEY` | Standard, avoids hardcoded keys |

Quark production uses Elasticsearch + Azure OpenAI for the same role; this
demo deliberately picks free, local equivalents to keep the focus on the
flow itself rather than infrastructure.

## Project layout

```
~/Desktop/rag-demo/
├── rag_demo.ipynb       # the 7-cell notebook
├── demo.md              # source text with markdown sections
├── chroma_db/            # persistent Chroma store (created on first run)
├── requirements.txt
├── .env.example         # VOYAGE_API_KEY=your_key_here
├── .gitignore           # ignores .env, chroma_db/, .venv/, .ipynb_checkpoints/
└── docs/superpowers/specs/2026-05-05-rag-demo-design.md  # this file
```

## Notebook structure

| Cell | Purpose |
|---|---|
| 1 | Setup — imports, load `VOYAGE_API_KEY`, define paths |
| 2 | Load `demo.md`, preview content |
| 3 | **Step 1** — chunk by markdown heading |
| 4 | **Step 2** — embed each chunk via Voyage AI (`input_type="document"`) |
| 5 | **Step 3** — create persistent Chroma collection, add embeddings + chunk text + metadata |
| 6 | **Step 4** — define a user question, embed it (`input_type="query"`) |
| 7 | **Step 5** — query Chroma for top-2 chunks, print results with similarity scores |

A markdown cell at the top of the notebook contains the run instructions
(virtualenv, requirements install, .env setup, jupyter launch).

## The five steps in detail

### Step 1 — Chunk by section

Read `demo.md`. Walk lines top-to-bottom. A line starting with `#`, `##`, or
`###` opens a new chunk. The chunk's text is the heading line plus all
subsequent lines until the next heading (or end of file).

Output: `list[dict]` of the form
`{"id": "section-N", "heading": "...", "text": "..."}`.

If the file contains zero headings, raise a clear error — proceeding would
silently produce an empty store and confusing query results.

No external chunking library; this is ~15 lines of pure Python.

### Step 2 — Embed each chunk

Single batch call to Voyage AI:

```python
client = voyageai.Client()  # picks up VOYAGE_API_KEY from env
result = client.embed(
    texts=[c["text"] for c in chunks],
    model="voyage-3-lite",
    input_type="document",
)
embeddings = result.embeddings  # list[list[float]], 512-dim each
```

Voyage's embeddings are asymmetric: `input_type="document"` for stored
chunks, `input_type="query"` for queries. Using both correctly is part of
what makes Voyage retrieval-quality competitive, so the demo uses both.

### Step 3 — Create vector DB and add embeddings

```python
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection(name="demo")
collection.add(
    ids=[c["id"] for c in chunks],
    embeddings=embeddings,
    documents=[c["text"] for c in chunks],
    metadatas=[{"heading": c["heading"]} for c in chunks],
)
```

`get_or_create_collection` is idempotent. Re-running the cell after editing
`demo.md` requires explicitly clearing the collection — a commented-out
`chroma.delete_collection("demo")` line is included for this.

### Step 4 — Embed the question

Same Voyage client, `input_type="query"`:

```python
question = "How do plants make food?"
q_emb = client.embed(
    texts=[question],
    model="voyage-3-lite",
    input_type="query",
).embeddings[0]
```

### Step 5 — Retrieve top-2 chunks

```python
results = collection.query(query_embeddings=[q_emb], n_results=2)
```

Print each of the two results with its heading, cosine distance, and a
200-character text preview. Lower distance = more relevant.

## Demo content (`demo.md`)

Four sections on deliberately disjoint topics so retrieval differences are
visually unambiguous:

1. Photosynthesis
2. The Roman Aqueducts
3. Bitcoin Mining
4. Honeybee Communication

Each section ~3–5 sentences. A query like "How do plants make food?" should
score Photosynthesis far higher than the other three; "What's a waggle
dance?" should pick Honeybee Communication.

## Error handling

Only at system boundaries. No defensive try/except inside the pipeline.

- Setup cell raises if `VOYAGE_API_KEY` is missing.
- Step 1 raises if `demo.md` has zero headings.
- All other failures (network, Voyage rate limits, Chroma I/O) are allowed
  to surface as native exceptions so they're visible.

## Persistence test ("some time later")

After running cells 1–5, restart the Jupyter kernel and run cells 1, 6, 7
only. Chroma reopens `./chroma_db` from disk and retrieval still works
without re-indexing. This makes step 4's temporal split real rather than
nominal.

## Dependencies (`requirements.txt`)

```
voyageai>=0.2.0
chromadb>=0.5.0
python-dotenv>=1.0.0
jupyter>=1.0.0
```

## Run instructions

```bash
cd ~/Desktop/rag-demo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "VOYAGE_API_KEY=your_key_here" > .env  # key from voyageai.com
jupyter notebook rag_demo.ipynb
```

Run cells top-to-bottom. To simulate the "some time later" gap, restart
the kernel after cell 5 and run cells 1, 6, 7.
