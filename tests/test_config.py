"""Smoke test: confirms the project imports and the folder layout exists."""
from src import config


def test_seed_is_fixed():
    assert config.RANDOM_SEED == 42


def test_project_folders_exist():
    for folder in [config.DATA_RAW, config.DATA_PROCESSED, config.MODELS_DIR, config.FIGURES_DIR]:
        assert folder.is_dir(), f"missing folder: {folder}"
