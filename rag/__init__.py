"""The `rag` package: one module per RAG concept.

    config      -> Settings from .env                   (all steps)
    loader      -> files to Documents                   (step 01)
    chunker     -> Documents to Chunks                  (step 02)
    embeddings  -> text to vectors via Ollama           (step 03)
    store       -> vectors into ChromaDB                (step 04)
    retriever   -> question to ranked Hits              (step 05)
    generator   -> Hits + question to grounded answer   (steps 06-07)

Read them in that order: it is the order data flows through the pipeline.
"""

import sys

# Windows consoles often default to a legacy code page (cp1252). Documents can contain
# any Unicode character, and printing one would crash a demo mid-session. Forcing UTF-8
# with "replace" means the worst case is a "?" on screen, never a traceback.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
