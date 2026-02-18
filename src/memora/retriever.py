import numpy as np
from rank_bm25 import BM25Okapi
from memora.store import Store
from memora.config import settings


class HybridRetriever:
    def __init__(self, store: Store):
        self.store = store

    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        k = k or settings.top_k
        if not self.store.meta:
            return []

        # FAISS semantic search
        sem = self.store.search(query, k=k * 2)

        # BM25 keyword search
        corpus = [m["text"].lower().split() for m in self.store.meta]
        bm25 = BM25Okapi(corpus)
        bm_scores = bm25.get_scores(query.lower().split())
        top_bm = np.argsort(bm_scores)[::-1][: k * 2]
        bm_results = [
            {**self.store.meta[i], "score": float(bm_scores[i])}
            for i in top_bm
            if bm_scores[i] > 0
        ]

        # Reciprocal Rank Fusion (k=60)
        scored: dict[int, float] = {}
        for rank, doc in enumerate(sem):
            scored[doc["id"]] = scored.get(doc["id"], 0) + 1 / (rank + 60)
        for rank, doc in enumerate(bm_results):
            scored[doc["id"]] = scored.get(doc["id"], 0) + 1 / (rank + 60)

        top_ids = sorted(scored, key=scored.get, reverse=True)[:k]
        meta_map = {m["id"]: m for m in self.store.meta}
        return [{**meta_map[i], "score": scored[i]} for i in top_ids if i in meta_map]
