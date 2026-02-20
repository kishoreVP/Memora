"""
Alternative Store implementation with lazy index rebuild.

This version marks removals but defers the expensive rebuild until:
1. Next search operation (when we need accurate results)
2. User explicitly runs 'memora compact'

This makes remove instant but searches may be slightly slower until compaction.
"""

import json
import time
import hashlib
import numpy as np
import faiss
from pathlib import Path
from memora.config import settings


class LazyStore:
    """Store with deferred index rebuilding for faster removals."""
    
    def __init__(self):
        settings.ensure_dirs()
        self.chunks_dir = settings.data_dir / "chunks"
        self.chunks_dir.mkdir(exist_ok=True)
        self.meta: list[dict] = []
        self.file_hashes: dict[str, str] = {}
        self._chunk_cache: dict[int, str] = {}
        self.index: faiss.Index | None = None
        self._needs_rebuild = False
        self._load()

    def _load(self):
        if settings.meta_path.exists():
            data = json.loads(settings.meta_path.read_text())
            self.meta = data.get("chunks", [])
            self.file_hashes = data.get("hashes", {})
            self._needs_rebuild = data.get("needs_rebuild", False)

        if settings.index_path.exists():
            self.index = faiss.read_index(str(settings.index_path))
        else:
            from memora.store import _dim
            self.index = faiss.IndexFlatIP(_dim())

        # Load chunks into cache
        for m in self.meta:
            path = self.chunks_dir / f"{m['id']}.txt"
            if path.exists():
                self._chunk_cache[m["id"]] = path.read_text(encoding="utf-8")

    def remove_fast(self, source: str) -> int:
        """Fast remove - just marks for rebuild, doesn't rebuild immediately."""
        removed_ids = [m["id"] for m in self.meta if m["source"] == source]
        removed = len(removed_ids)
        
        if removed == 0:
            return 0

        # Remove from cache and disk
        for chunk_id in removed_ids:
            self._chunk_cache.pop(chunk_id, None)
            path = self.chunks_dir / f"{chunk_id}.txt"
            if path.exists():
                path.unlink()

        # Update metadata
        self.meta = [m for m in self.meta if m["source"] != source]
        self.file_hashes.pop(source, None)
        
        # Mark as needing rebuild instead of rebuilding now
        self._needs_rebuild = True
        
        # Save metadata with rebuild flag
        data = {
            "chunks": self.meta,
            "hashes": self.file_hashes,
            "needs_rebuild": True
        }
        settings.meta_path.write_text(json.dumps(data, indent=2))
        
        return removed

    def compact(self):
        """Explicitly rebuild the index to compact it."""
        if not self._needs_rebuild:
            return
        
        from memora.store import _dim, embed
        
        self.index = faiss.IndexFlatIP(_dim())
        if self.meta:
            texts = [self._chunk_cache[m["id"]] for m in self.meta]
            vecs = embed(texts)
            self.index.add(vecs)
        
        self._needs_rebuild = False
        faiss.write_index(self.index, str(settings.index_path))
        
        data = {
            "chunks": self.meta,
            "hashes": self.file_hashes,
            "needs_rebuild": False
        }
        settings.meta_path.write_text(json.dumps(data, indent=2))

    def search_with_auto_compact(self, query: str, k: int = 5):
        """Search - auto-compacts if needed."""
        if self._needs_rebuild:
            self.compact()
        
        from memora.store import embed
        vec = embed([query])
        scores, indices = self.index.search(vec, k)
        
        results = []
        for j, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.meta):
                m = self.meta[idx]
                results.append({
                    **m,
                    "text": self._chunk_cache[m["id"]],
                    "score": float(scores[0][j])
                })
        return results
