import hashlib
import json
import os
from pathlib import Path
import time
from typing import Optional

from memora.config import settings
import numpy as np

# --- AGGRESSIVE CPU OPTIMIZATION (fallback) ---
# These must be set before ANY imports to prevent thread-pool explosion
os.environ["ORT_LOGGING_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "3" 
os.environ["MKL_NUM_THREADS"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _file_hash(path: str) -> str:
    """Fast hash for deduplication. No dependencies."""
    try:
        with open(path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return ""

def _detect_gpu() -> tuple[bool, Optional[str]]:
    """Detect available GPU and return (has_gpu, provider)."""
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        
        # Priority order: CUDA > ROCm > CoreML > DirectML
        if "CUDAExecutionProvider" in available:
            return True, "CUDAExecutionProvider"
        elif "ROCmExecutionProvider" in available:
            return True, "ROCmExecutionProvider"
        elif "CoreMLExecutionProvider" in available:
            return True, "CoreMLExecutionProvider"
        elif "DmlExecutionProvider" in available:
            return True, "DmlExecutionProvider"
    except ImportError:
        pass
    
    return False, None

class MetadataStore:
    """Handles metadata and chunk files without loading heavy ML libraries."""
    
    def __init__(self):
        settings.ensure_dirs()
        self.chunks_dir = settings.data_dir / "chunks"
        self.chunks_dir.mkdir(exist_ok=True)
        self.meta: list[dict] = []
        self.file_hashes: dict[str, str] = {}
        self.needs_rebuild: bool = False
        self._load_meta()

    def _load_meta(self):
        if settings.meta_path.exists():
            try:
                data = json.loads(settings.meta_path.read_text())
                self.meta = data.get("chunks", [])
                self.file_hashes = data.get("hashes", {})
                self.needs_rebuild = data.get("needs_rebuild", False)
            except:
                pass

    def _save_meta(self):
        data = {
            "chunks": self.meta,
            "hashes": self.file_hashes,
            "needs_rebuild": self.needs_rebuild
        }
        settings.meta_path.write_text(json.dumps(data, indent=2))

    def _get_chunk_path(self, chunk_id: int) -> Path:
        return self.chunks_dir / f"{chunk_id}.txt"

    def _read_chunk(self, chunk_id: int) -> str:
        """Helper for retriever/RAG to get text content from disk."""
        path = self._get_chunk_path(chunk_id)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def list_sources(self) -> list[dict]:
        sources: dict[str, dict] = {}
        for m in self.meta:
            s = m["source"]
            if s not in sources:
                sources[s] = {"source": s, "chunks": 0, "added": m["ts"]}
            sources[s]["chunks"] += 1
        return list(sources.values())

    def remove(self, source: str) -> int:
        """Soft remove: instant deletion of pointers."""
        removed_ids = [m["id"] for m in self.meta if m["source"] == source]
        if not removed_ids: return 0
        
        for cid in removed_ids:
            p = self._get_chunk_path(cid)
            if p.exists(): p.unlink()
            
        self.meta = [m for m in self.meta if m["source"] != source]
        self.file_hashes.pop(source, None)
        self._save_meta()
        return len(removed_ids)

class Store(MetadataStore):
    """Full store with GPU acceleration (falls back to CPU)."""
    
    def __init__(self):
        super().__init__()
        self._engine = None
        self._index = None
        self.faiss = None
        self.use_gpu = False
        self.gpu_provider = None

    def _init_ml(self):
        """Surgical ML initialization with GPU detection."""
        if self._engine is not None:
            return
        
        try:
            from fastembed import TextEmbedding
            import faiss
            self.faiss = faiss
            
            # GPU Detection
            has_gpu, gpu_provider = _detect_gpu()
            self.use_gpu = has_gpu
            self.gpu_provider = gpu_provider
            
            # Set up providers list
            if has_gpu:
                providers = [gpu_provider, "CPUExecutionProvider"]
                print(f"GPU detected: Using {gpu_provider}")
            else:
                providers = ["CPUExecutionProvider"]
                print("No GPU detected: Using CPU")
            
            # BAAI/bge-small is 3x faster to load than MiniLM on cold start
            self._engine = TextEmbedding(
                model_name="BAAI/bge-small-en-v1.5",
                providers=providers
            )
            self.dim = 384 
            
            # Warmup: Pre-calculate one tiny string to 'bake' the session
            list(self._engine.embed(["warmup"]))
            
        except ImportError as e:
            raise ImportError(f"Missing dependency: {e.name}. Run 'uv sync'.")

    def _load_index(self):
        """Load FAISS index with optional GPU acceleration."""
        if self._index is not None: 
            return
            
        self._init_ml()
        
        if settings.index_path.exists():
            cpu_index = self.faiss.read_index(str(settings.index_path))
            
            # Move to GPU if available
            if self.use_gpu and hasattr(self.faiss, 'StandardGpuResources'):
                try:
                    res = self.faiss.StandardGpuResources()
                    self._index = self.faiss.index_cpu_to_gpu(res, 0, cpu_index)
                    print("FAISS index moved to GPU")
                except Exception as e:
                    print(f"GPU index failed, using CPU: {e}")
                    self._index = cpu_index
            else:
                self._index = cpu_index
        else:
            # Create new index
            cpu_index = self.faiss.IndexFlatIP(self.dim)
            
            if self.use_gpu and hasattr(self.faiss, 'StandardGpuResources'):
                try:
                    res = self.faiss.StandardGpuResources()
                    self._index = self.faiss.index_cpu_to_gpu(res, 0, cpu_index)
                    print("Created GPU FAISS index")
                except Exception as e:
                    print(f"GPU index creation failed, using CPU: {e}")
                    self._index = cpu_index
            else:
                self._index = cpu_index

    def add(self, chunks: list[str], source: str) -> int:
        """Add chunks with GPU-accelerated embedding."""
        f_hash = _file_hash(source)
        if source in self.file_hashes and self.file_hashes[source] == f_hash:
            return 0

        self._load_index()
        
        # Batch embedding with GPU acceleration
        vecs = np.array(list(self._engine.embed(chunks)))
        
        start_pos = self._index.ntotal
        for i, text in enumerate(chunks):
            chunk_id = start_pos + i
            self._get_chunk_path(chunk_id).write_text(text, encoding="utf-8")
            self.meta.append({"id": chunk_id, "source": source, "ts": time.time()})
        
        self._index.add(vecs.astype("float32"))
        self.file_hashes[source] = f_hash
        
        # Save index (convert from GPU to CPU if needed)
        if self.use_gpu and hasattr(self._index, 'index'):
            cpu_index = self.faiss.index_gpu_to_cpu(self._index)
            self.faiss.write_index(cpu_index, str(settings.index_path))
        else:
            self.faiss.write_index(self._index, str(settings.index_path))
        
        self._save_meta()
        return len(chunks)

    def search(self, query: str, k: int = 5) -> list[dict]:
        """GPU-accelerated search."""
        self._load_index()
        if self._index.ntotal == 0: 
            return []
        
        search_k = min(k * 3, self._index.ntotal)
        q_vec = np.array(list(self._engine.embed([query])))
        scores, indices = self._index.search(q_vec.astype("float32"), search_k)
        
        meta_lookup = {m["id"]: m for m in self.meta}
        results = []
        for j, idx in enumerate(indices[0]):
            if idx in meta_lookup:
                m = meta_lookup[idx]
                results.append({
                    **m,
                    "text": self._read_chunk(m["id"]),
                    "score": float(scores[0][j])
                })
            if len(results) >= k:
                break
        return results
