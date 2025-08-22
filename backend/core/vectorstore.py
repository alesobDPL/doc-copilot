# backend/core/vectorstore.py
import os

BACKEND = os.getenv("VSTORE_BACKEND", "mini").lower()  # "mini" | "chroma" | "qdrant"

if BACKEND == "chroma":
    from .vstore_chroma import (
        upsert_chunks, query, load_collection, clear_collection
    )
else:
    # tu mini-store actual
    from .vstore_mini import (
        upsert_chunks, query, load_collection, clear_collection
    )


# elif BACKEND == "qdrant":
#    # si más adelante agregas Qdrant:
#    from .vstore_qdrant import (
#        upsert_chunks, query, load_collection, clear_collection
#    ) 
#