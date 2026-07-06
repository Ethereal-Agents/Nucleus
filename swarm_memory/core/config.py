import os

DB_PATH = os.getenv("SWARM_MEMORY_DB_PATH", "swarm_memory.db")
EMBED_DIM = int(os.getenv("SWARM_MEMORY_EMBED_DIM", "768"))
EMBED_MODEL = os.getenv("SWARM_MEMORY_EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")
RRF_K = int(os.getenv("SWARM_MEMORY_RRF_K", "60"))
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3-haiku")
