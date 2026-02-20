import json
import os
import time
import logging
import warnings
import numpy as np
import faiss
import hashlib
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from memora.config import settings

_model = None

def _get_model():
    global _model
    if _model is None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        logging.getLogger("torch").setLevel(logging.ERROR)

        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    return _get_model().encode(texts, normalize_embeddings=True, show_progress_bar=False).astype("float32")


def _dim() -> int:
    return _get_model().get_sentence_embedding_dimension()


def _file_hash(path: str) -> str:
    """Generate hash of file for deduplication."""
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


class Store:
    def __init__(self):
        settings.ensure_dirs()
        self.chunks_dir = settings.data_dir / "chunks"
        self.chunks_dir.mkdir(exist_ok=True)
        self.meta: list[dict] = []
        self.file_hashes: dict[str, str] = {}  # source -> hash
        self.index: faiss.Index | None = None
        self._load()

    def _load(self):
        if settings.meta_path.exists():
            data = json.loads(settings.meta_path.read_text())
            self.meta = data.get("chunks", [])
            self.file_hashes = data.get("hashes", {})
        
        if settings.index_path.exists():
            self.index = faiss.read_index(str(settings.index_path))
        else:
            # Use IVF index for better compression (train after 1000+ vectors)
            self.index = faiss.IndexFlatIP(_dim())

    def _save(self):
        # Save only metadata, not full text
        data = {
            "chunks": self.meta,
            "hashes": self.file_hashes
        }
        settings.meta_path.write_text(json.dumps(data, indent=2))
        faiss.write_index(self.index, str(settings.index_path))

    def _get_chunk_path(self, chunk_id: int) -> Path:
        """Get path to chunk text file."""
        return self.chunks_dir / f"{chunk_id}.txt"

    def _write_chunk(self, chunk_id: int, text: str):
        """Write chunk text to disk."""
        self._get_chunk_path(chunk_id).write_text(text, encoding="utf-8")

    def _read_chunk(self, chunk_id: int) -> str:
        """Read chunk text from disk."""
        path = self._get_chunk_path(chunk_id)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def add(self, chunks: list[str], source: str) -> int:
        # Check if file already indexed
        try:
            file_hash = _file_hash(source)
            if source in self.file_hashes and self.file_hashes[source] == file_hash:
                return 0  # Already indexed, skip
            self.file_hashes[source] = file_hash
        except:
            pass  # If hashing fails, proceed anyway

        vecs = embed(chunks)
        start = self.index.ntotal
        self.index.add(vecs)
        ts = time.time()
        
        for i, chunk in enumerate(chunks):
            chunk_id = start + i
            # Write chunk text to disk
            self._write_chunk(chunk_id, chunk)
            # Store only metadata
            self.meta.append({
                "id": chunk_id, 
                "source": source, 
                "ts": ts
            })
        
        self._save()
        
        # Convert to IVF if we have enough vectors
        if self.index.ntotal >= 1000 and isinstance(self.index, faiss.IndexFlatIP):
            self._convert_to_ivf()
        
        return len(chunks)

    def _convert_to_ivf(self):
        """Convert flat index to IVF for better performance."""
        print("Converting to IVF index for better performance...")
        nlist = min(100, self.index.ntotal // 10)  # Number of clusters
        quantizer = faiss.IndexFlatIP(_dim())
        new_index = faiss.IndexIVFFlat(quantizer, _dim(), nlist)
        
        # Train on existing vectors
        vectors = []
        for m in self.meta:
            text = self._read_chunk(m["id"])
            vectors.append(text)
        
        vecs = embed(vectors)
        new_index.train(vecs)
        new_index.add(vecs)
        
        self.index = new_index
        self._save()

    def search(self, query: str, k: int | None = None) -> list[dict]:
        k = min(k or settings.top_k, self.index.ntotal) if self.index.ntotal else 0
        if k == 0:
            return []
        
        # Set nprobe for IVF indexes
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = 10
        
        vec = embed([query])
        scores, ids = self.index.search(vec, k)
        
        results = []
        for j, i in enumerate(ids[0]):
            if i < len(self.meta):
                meta = self.meta[i]
                text = self._read_chunk(meta["id"])
                results.append({
                    **meta,
                    "text": text,
                    "score": float(scores[0][j])
                })
        return results

    def remove(self, source: str) -> int:
        keep = [m for m in self.meta if m["source"] != source]
        removed = len(self.meta) - len(keep)
        if removed == 0:
            return 0
        
        # Remove chunk files
        for m in self.meta:
            if m["source"] == source:
                chunk_path = self._get_chunk_path(m["id"])
                if chunk_path.exists():
                    chunk_path.unlink()
        
        # Remove from hashes
        self.file_hashes.pop(source, None)
        
        self.meta = keep
        self.index = faiss.IndexFlatIP(_dim())
        if keep:
            vectors = []
            for m in keep:
                text = self._read_chunk(m["id"])
                vectors.append(text)
            vecs = embed(vectors)
            self.index.add(vecs)
            for i, m in enumerate(keep):
                m["id"] = i
        self._save()
        return removed

    def list_sources(self) -> list[dict]:
        sources: dict[str, dict] = {}
        for m in self.meta:
            s = m["source"]
            if s not in sources:
                sources[s] = {"source": s, "chunks": 0, "added": m["ts"]}
            sources[s]["chunks"] += 1
        return list(sources.values())

    def stats(self) -> dict:
        sources = self.list_sources()
        return {
            "total_chunks": len(self.meta),
            "total_sources": len(sources),
            "index_size": self.index.ntotal,
        }
