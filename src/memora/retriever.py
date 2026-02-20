import numpy as np
import pickle
from rank_bm25 import BM25Okapi
from memora.store import Store
from memora.config import settings


class HybridRetriever:
    def __init__(self, store: Store):
        self.store = store
        self.bm25 = None
        self.bm25_path = settings.data_dir / "bm25.pkl"
        self._load_bm25()

    def _load_bm25(self):
        """Load cached BM25 or rebuild if needed."""
        if self.bm25_path.exists() and self.store.meta:
            try:
                with open(self.bm25_path, 'rb') as f:
                    self.bm25 = pickle.load(f)
                return
            except:
                pass
        self._rebuild_bm25()

    def _rebuild_bm25(self):
        """Rebuild and cache BM25 index."""
        if not self.store.meta:
            self.bm25 = None
            return
        
        corpus = []
        for m in self.store.meta:
            text = self.store._read_chunk(m["id"])
            corpus.append(text.lower().split())
        
        self.bm25 = BM25Okapi(corpus)
        
        with open(self.bm25_path, 'wb') as f:
            pickle.dump(self.bm25, f)

    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        k = k or settings.top_k
        if not self.store.meta:
            return []

        # FAISS semantic search
        sem = self.store.search(query, k=k * 2)

        # BM25 keyword search
        if self.bm25 is None:
            self._rebuild_bm25()
        
        if self.bm25 is None:
            return sem[:k]  # Fallback to just semantic
        
        bm_scores = self.bm25.get_scores(query.lower().split())
        top_bm = np.argsort(bm_scores)[::-1][: k * 2]
        bm_results = [
            {**self.store.meta[i], "score": float(bm_scores[i])}
            for i in top_bm
            if i < len(self.store.meta) and bm_scores[i] > 0
        ]

        # Reciprocal Rank Fusion
        scored: dict[int, float] = {}
        for rank, doc in enumerate(sem):
            scored[doc["id"]] = scored.get(doc["id"], 0) + 1 / (rank + 60)
        for rank, doc in enumerate(bm_results):
            scored[doc["id"]] = scored.get(doc["id"], 0) + 1 / (rank + 60)

        top_ids = sorted(scored, key=scored.get, reverse=True)[:k]
        
        # Fetch full text for results
        results = []
        for chunk_id in top_ids:
            for m in self.store.meta:
                if m["id"] == chunk_id:
                    text = self.store._read_chunk(chunk_id)
                    results.append({
                        **m,
                        "text": text,
                        "score": scored[chunk_id]
                    })
                    break
        
        return results
