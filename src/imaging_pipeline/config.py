"""Project-wide paths and constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "nuclei_dataset"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
TEST_CORRUPTED_DIR = DATA_DIR / "test_corrupted"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
RECORDS_DIR = OUTPUT_DIR / "records"

for d in [FIGURE_DIR, METRICS_DIR, RECORDS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Image geometry
IMAGE_SIZE = (256, 256)

# Ollama models
VLM_MODEL = "llama3.2-vision"
TEXT_LLM_MODEL = "llama3.2"

# Modality/dataset description (fixed facts about this assignment's data,
# used to keep prompts and report text consistent)
MODALITY_NAME = "fluorescence microscopy (DAPI-style stained nuclei)"
