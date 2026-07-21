"""
swarm_memory/core/config.py

Central configuration for the SwarmMemory system.
All values are read from environment variables with sensible defaults,
making the system easy to configure for both local dev and production.

Usage:
    from swarm_memory.core import config
    conn = sqlite3.connect(config.DB_PATH)
"""

import os

from dotenv import load_dotenv

load_dotenv(override=True)

# ── Logging ─────────────────────────────────────────────────────────────────

# Logging level for the application.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# ── Database ────────────────────────────────────────────────────────────────

# Path to the SQLite database file.
# For local dev this can be a relative path; for a shared SSE server,
# use an absolute path on the host machine.
DB_PATH = os.getenv("SWARM_MEMORY_DB_PATH", "swarm_memory.db")

# ── Embedding model ─────────────────────────────────────────────────────────

# HuggingFace model identifier for sentence-transformers.
# nomic-embed-text-v1.5 is a 137M-param model with Matryoshka Representation
# Learning (MRL), meaning you can truncate its 768-dim output to 256 dims
# and still get ~95% of the quality at 3× less storage.
#
# TODO: Migrate to ONNX Runtime for 2-4× faster CPU inference once the
#       pipeline is validated. nomic-embed-text-v1.5 has a pre-exported
#       ONNX model on HuggingFace: nomic-ai/nomic-embed-text-v1.5-ONNX
EMBED_MODEL = os.getenv(
    "SWARM_MEMORY_EMBED_MODEL",
    "nomic-ai/nomic-embed-text-v1.5",
)

# Number of embedding dimensions to keep (Matryoshka truncation).
# 768 = full quality (default). Set to 256 for 3× storage savings (~5% quality loss).
EMBED_DIM = int(os.getenv("SWARM_MEMORY_EMBED_DIM", "768"))

# ── Retrieval ────────────────────────────────────────────────────────────────

# Reciprocal Rank Fusion constant.
# Controls weight distribution between top vs. lower-ranked results.
# Higher k → more uniform; lower k → top-heavy. 60 is the standard default.
RRF_K = int(os.getenv("SWARM_MEMORY_RRF_K", "60"))

# Weights for dense and BM25 search in RRF fusion.
# Dense is primary (1.0), BM25 is secondary/booster (0.8).
DENSE_WEIGHT = float(os.getenv("SWARM_MEMORY_DENSE_WEIGHT", "1.0"))
BM25_WEIGHT = float(os.getenv("SWARM_MEMORY_BM25_WEIGHT", "0.8"))

# Fact extraction settings
DEFAULT_FACT_TYPE = "insight"
MIN_CONFIDENCE_THRESHOLD = 0.5
FACT_WORD_THRESHOLD = 150  # Split facts longer than this many words into smaller independent chunks

# Confidence decay per day (0.01 = 1% per day).
# Applied to fact relevance scores to give recency bias:
#   adjusted_score = score * max(DECAY_FLOOR, 1.0 - age_days * DECAY_RATE)
CONFIDENCE_DECAY_RATE = float(os.getenv("SWARM_MEMORY_CONFIDENCE_DECAY_RATE", "0.01"))

# Minimum score multiplier so old-but-useful facts never vanish entirely.
CONFIDENCE_DECAY_FLOOR = float(os.getenv("SWARM_MEMORY_CONFIDENCE_DECAY_FLOOR", "0.5"))

# ── Session deduplication ────────────────────────────────────────────────────

# How long (seconds) a session's "seen fact IDs" set is kept alive in memory.
# After this time, the session is evicted. This is a safety net for agents
# that crash without calling memory_end_run().
# Default: 7200 = 2 hours.
SESSION_TTL_SECONDS = int(os.getenv("SWARM_MEMORY_SESSION_TTL", "7200"))

# ── LLM (LiteLLM) ────────────────────────────────────────────────────────────

# Used by the core LLMService (e.g. for supersession detection).
# Since we use LiteLLM, you can prefix with provider (e.g. openrouter/..., openai/...)
LLM_MODEL = os.getenv("SWARM_MEMORY_LLM_MODEL", "openrouter/xiaomi/mimo-v2.5-pro")
