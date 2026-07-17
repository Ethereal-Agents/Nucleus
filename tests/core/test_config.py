import os
import importlib
from unittest import mock
import pytest

def reload_config():
    import swarm_memory.core.config as config
    importlib.reload(config)
    return config

@pytest.fixture(autouse=True)
def mock_dotenv():
    # Prevent dotenv from overriding our mocked os.environ
    with mock.patch("dotenv.load_dotenv"):
        yield

def test_default_values():
    with mock.patch.dict(os.environ, {}, clear=True):
        config = reload_config()
        assert config.LOG_LEVEL == "INFO"
        assert config.DB_PATH == "swarm_memory.db"
        assert config.EMBED_MODEL == "nomic-ai/nomic-embed-text-v1.5"
        assert config.EMBED_DIM == 768
        assert config.RRF_K == 60
        assert config.DENSE_WEIGHT == 1.0
        assert config.BM25_WEIGHT == 0.8
        assert config.CONFIDENCE_DECAY_RATE == 0.01
        assert config.CONFIDENCE_DECAY_FLOOR == 0.5
        assert config.SESSION_TTL_SECONDS == 7200
        assert config.LLM_MODEL == "openrouter/xiaomi/mimo-v2.5-pro"

def test_db_path_from_env():
    with mock.patch.dict(os.environ, {"SWARM_MEMORY_DB_PATH": "/custom/path.db"}):
        config = reload_config()
        assert config.DB_PATH == "/custom/path.db"

def test_embed_dim_from_env():
    with mock.patch.dict(os.environ, {"SWARM_MEMORY_EMBED_DIM": "256"}):
        config = reload_config()
        assert config.EMBED_DIM == 256

def test_rrf_k_from_env():
    with mock.patch.dict(os.environ, {"SWARM_MEMORY_RRF_K": "40"}):
        config = reload_config()
        assert config.RRF_K == 40

def test_weights_from_env():
    with mock.patch.dict(os.environ, {"SWARM_MEMORY_DENSE_WEIGHT": "2.0", "SWARM_MEMORY_BM25_WEIGHT": "1.5"}):
        config = reload_config()
        assert config.DENSE_WEIGHT == 2.0
        assert config.BM25_WEIGHT == 1.5

def test_decay_params_from_env():
    with mock.patch.dict(os.environ, {"SWARM_MEMORY_CONFIDENCE_DECAY_RATE": "0.05", "SWARM_MEMORY_CONFIDENCE_DECAY_FLOOR": "0.2"}):
        config = reload_config()
        assert config.CONFIDENCE_DECAY_RATE == 0.05
        assert config.CONFIDENCE_DECAY_FLOOR == 0.2

def test_llm_model_from_env():
    with mock.patch.dict(os.environ, {"SWARM_MEMORY_LLM_MODEL": "gpt-4"}):
        config = reload_config()
        assert config.LLM_MODEL == "gpt-4"

def test_session_ttl_from_env():
    with mock.patch.dict(os.environ, {"SWARM_MEMORY_SESSION_TTL": "3600"}):
        config = reload_config()
        assert config.SESSION_TTL_SECONDS == 3600
