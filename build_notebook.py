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

CELL_5B_REOPEN = '''\
# Cell 5b — Persistence-test helper: reopen the existing collection from disk.
#
# After running cells 1–5 once, restart the Jupyter kernel (Kernel → Restart),
# then run cell 1, this cell, and cells 6–7. The `collection.count()` here
# proves the on-disk Chroma store survived the restart — no re-embedding
# needed.
chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma.get_or_create_collection(name=COLLECTION_NAME)
print(f"Reopened collection {COLLECTION_NAME!r} with {collection.count()} items.")
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
        nbf.v4.new_code_cell(CELL_5B_REOPEN),
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
