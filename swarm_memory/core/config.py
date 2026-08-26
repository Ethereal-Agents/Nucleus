"""
swarm_memory/core/config.py

Central configuration for the SwarmMemory system.
All values are read from environment variables with sensible defaults,
making the system easy to configure for both local dev and production.

Usage:
    from swarm_memory.core import config
    conn = sqlite3.connect(config.DB_PATH)
"""

import math
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

# Model identifier for fastembed (ONNX runtime).
# nomic-embed-text-v1.5 is a 137M-param model with Matryoshka Representation
# Learning, meaning its 768-dim output can be safely truncated down to 256
# while retaining 95% of performance.
#
# Currently running via ONNX for minimal cold starts and low CPU latency.
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

# Minimum cosine similarity for a fact to be included in search results.
# Only facts whose embedding is at least this similar to the query are returned.
# 0.60 = 60% cosine similarity (recommended). Set to 0.0 to disable the gate.
#
# For unit-norm embeddings (nomic-embed-text), cosine similarity maps to L2
# distance via:  distance = sqrt(2 - 2 * cosine_similarity)
# 0.60 similarity → max_distance ≈ 0.8944
RETRIEVAL_MIN_SIMILARITY = float(os.getenv("SWARM_MEMORY_RETRIEVAL_MIN_SIMILARITY", "0.60"))
# Convert cosine similarity to sqlite-vec L2 distance max.
# sqlite-vec uses L2 distance for exact KNN. Since embeddings are L2 normalized,
# L2 distance = sqrt(2 - 2 * cosine_similarity).
# For similarity >= 0.60, L2 distance must be <= sqrt(2 - 2 * 0.60) = sqrt(0.8) ≈ 0.8944.
RETRIEVAL_MAX_DISTANCE = math.sqrt(2.0 - 2.0 * RETRIEVAL_MIN_SIMILARITY)

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
LLM_MODEL = os.getenv("SWARM_MEMORY_LLM_MODEL", "openrouter/deepseek/deepseek-v4-flash-0731")
