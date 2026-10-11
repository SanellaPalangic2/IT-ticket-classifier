"""Project-wide settings: the random seed and file paths.

Every script imports from here, so there is exactly one place to change
the seed or a path.
"""
from pathlib import Path

# The single random seed used everywhere (data splits, cross-validation,
# models). Fixing it means anyone who reruns the scripts gets the same numbers.
RANDOM_SEED = 42

# Paths are built from this file's location, so the scripts work no matter
# which folder you run them from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
