"""Machine-specific locations. Override with environment variables."""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASAP = Path(os.environ.get("SUBHMM_ASAP", "/disk1/jaehoon/dataset_store/asap"))
MAESTRO_ZIP = Path(os.environ.get("SUBHMM_MAESTRO_ZIP", "/disk1/jaehoon/dataset_store/maestro/maestro-v3.0.0.zip"))
# Explore This checkout (its explore_this package reads the feature store) and its 50 fps log-mel store,
# both only needed for the optional mel50 onset source
EXPLORE_THIS_REPO = Path(os.environ.get("SUBHMM_EXPLORE_THIS", "/home/sogang/jaehoon/explore_this"))
# 50 fps log-mel store used by Explore This (only for the optional mel50 onset source)
FEATURES = Path(os.environ.get("SUBHMM_FEATURES", "/home/sogang/jaehoon/skip_that_beat/explore_this_experiments/runs/beat_this_full/data/audio/spectrograms/asap.npz"))
# Per-performance alignment/audio/split facts derived from ASAP + MAESTRO metadata
INVENTORY = HERE / "data/performance_inventory.csv"
