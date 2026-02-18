import json
import os
import time
import logging
import warnings
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from memora.config import settings

_model = None

def _get_model():
    global _model
    if _model is None:
        # Suppress all warnings before importing/loading model
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


class Store:
    def __init__(self):
        settings.ensure_dirs()
        self.meta: list[dict] = []
        self.index: faiss.IndexFlatIP | None = None
        self._load()

    def _load(self):
        if settings.meta_path.exists():
            self.meta = json.loads(settings.meta_path.read_text())
        if settings.index_path.exists():
            self.index = faiss.read_index(str(settings.index_path))
        if self.index is None:
            self.index = faiss.IndexFlatIP(_dim())

    def _save(self):
        settings.meta_path.write_text(json.dumps(self.meta, indent=2))
        faiss.write_index(self.index, str(settings.index_path))

    def add(self, chunks: list[str], source: str) -> int:
        vecs = embed(chunks)
        start = self.index.ntotal
        self.index.add(vecs)
        ts = time.time()
        for i, chunk in enumerate(chunks):
            self.meta.append({"id": start + i, "source": source, "text": chunk, "ts": ts})
        self._save()
        return len(chunks)

    def search(self, query: str, k: int | None = None) -> list[dict]:
        k = min(k or settings.top_k, self.index.ntotal) if self.index.ntotal else 0
        if k == 0:
            return []
        vec = embed([query])
        scores, ids = self.index.search(vec, k)
        return [
            {**self.meta[i], "score": float(scores[0][j])}
            for j, i in enumerate(ids[0]) if i < len(self.meta)
        ]

    def remove(self, source: str) -> int:
        keep = [m for m in self.meta if m["source"] != source]
        removed = len(self.meta) - len(keep)
        if removed == 0:
            return 0
        self.meta = keep
        self.index = faiss.IndexFlatIP(_dim())
        if keep:
            vecs = embed([m["text"] for m in keep])
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
