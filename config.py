import os
from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).resolve().parent

# Dataset Directories
DATASET_DIR = Path(os.getenv("DATASET_DIR", str(BASE_DIR / "Dataset" if (BASE_DIR / "Dataset").exists() else BASE_DIR / "RBI-GUIDELINES")))
PERSIST_DIRECTORY = BASE_DIR / ".chroma_db"
SPARSE_INDEX_PATH = BASE_DIR / ".bm25_index.pkl"

# Model Configurations (CPU Optimized)
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # 384-dim, fast & accurate on CPU
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Small CPU reranker

# Generation Configurations (Ollama CPU model)
OLLAMA_MODEL_NAME = "qwen2.5:7b-instruct-q4_K_M"
OLLAMA_FALLBACK_MODEL = "phi3.5:latest"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Chunking Parameters
TARGET_CHUNK_SIZE = 600      # Target token/word length per chunk
CHUNK_OVERLAP = 100          # Overlap length for sentence boundary splitting
MAX_CHUNK_SIZE = 1000        # Hard ceiling for chunks

# Retrieval & Hybrid Parameters
TOP_K_DENSE = 15
TOP_K_SPARSE = 15
TOP_K_HYBRID = 15
TOP_K_RERANKED = 5           # Final context chunks fed into LLM
RRF_K = 60                   # Reciprocal Rank Fusion constant

# Hugging Face Token (if needed for gated models)
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN", os.getenv("HF_TOKEN", ""))
if HF_TOKEN:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
