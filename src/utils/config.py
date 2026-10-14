"""
Dysarthria AI — Central Configuration

All project paths, preprocessing parameters, and label mappings are defined here.
Uses pathlib for cross-platform path handling. No hard-coded absolute paths.
"""

from pathlib import Path

# ============================================================
# Project Paths
# ============================================================

# Project root: the directory containing src/, data/, notebooks/, etc.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Dataset paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"

# UASpeech dataset root (inside raw/)
UASPEECH_ROOT = RAW_DATA_DIR / "UASpeech"
UASPEECH_AUDIO_DIR = UASPEECH_ROOT / "audio" / "noisereduce"
UASPEECH_DOC_DIR = UASPEECH_ROOT / "doc"
UASPEECH_MLF_DIR = UASPEECH_ROOT / "mlf"

# Output paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
LOGS_DIR = OUTPUTS_DIR / "logs"
RESULTS_DIR = OUTPUTS_DIR / "results"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"

# Metadata file paths
METADATA_CSV = METADATA_DIR / "uaspeech_metadata.csv"
TRAIN_CSV = METADATA_DIR / "train_metadata.csv"
VAL_CSV = METADATA_DIR / "val_metadata.csv"
TEST_CSV = METADATA_DIR / "test_metadata.csv"

# ============================================================
# Label Mapping
# ============================================================

# UASpeech convention:
#   Speakers with 'C' prefix (e.g., CF02, CM04) → Control
#   Speakers without 'C' prefix (e.g., F02, M04) → Dysarthric
LABEL_CONTROL = 0
LABEL_DYSARTHRIC = 1

LABEL_MAP = {
    LABEL_CONTROL: "Control",
    LABEL_DYSARTHRIC: "Dysarthric",
}

# ============================================================
# Audio Preprocessing Parameters
# ============================================================

# Target sampling rate for preprocessing
# UASpeech is recorded at 16 kHz — will be verified during exploration
TARGET_SAMPLE_RATE = 16000

# Mel Spectrogram parameters
N_MELS = 128          # Number of Mel frequency bands
N_FFT = 2048          # FFT window size
HOP_LENGTH = 512      # Hop length between frames

# Fixed duration for padding/cropping (in seconds)
# Will be refined after inspecting duration distribution
MAX_DURATION_SEC = 5.0

# Computed from duration and hop length
# target_frames = int(TARGET_SAMPLE_RATE * MAX_DURATION_SEC / HOP_LENGTH) + 1
# This gives the time dimension of the fixed-size Mel spectrogram
TARGET_FRAMES = int(TARGET_SAMPLE_RATE * MAX_DURATION_SEC / HOP_LENGTH) + 1

# Final tensor shape for CNN input: (n_mels, target_frames, 1)
INPUT_SHAPE = (N_MELS, TARGET_FRAMES, 1)

# ============================================================
# Data Split Configuration
# ============================================================

# Standard UASpeech block-based split
TRAIN_BLOCKS = ["B1", "B3"]
TEST_BLOCKS = ["B2"]

# Validation: fraction of training speakers held out for validation
VAL_SPEAKER_FRACTION = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================
# TensorFlow Pipeline Parameters
# ============================================================

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000
PREFETCH_BUFFER = 2  # tf.data.AUTOTUNE can also be used


def print_config():
    """Print current configuration for verification."""
    print("=" * 60)
    print("Dysarthria AI — Configuration")
    print("=" * 60)
    print(f"Project Root:       {PROJECT_ROOT}")
    print(f"Raw Data Dir:       {RAW_DATA_DIR}")
    print(f"UASpeech Audio Dir: {UASPEECH_AUDIO_DIR}")
    print(f"Metadata Dir:       {METADATA_DIR}")
    print(f"Figures Dir:        {FIGURES_DIR}")
    print()
    print(f"Target Sample Rate: {TARGET_SAMPLE_RATE} Hz")
    print(f"N Mels:             {N_MELS}")
    print(f"N FFT:              {N_FFT}")
    print(f"Hop Length:         {HOP_LENGTH}")
    print(f"Max Duration:       {MAX_DURATION_SEC} sec")
    print(f"Target Frames:      {TARGET_FRAMES}")
    print(f"Input Shape:        {INPUT_SHAPE}")
    print()
    print(f"Train Blocks:       {TRAIN_BLOCKS}")
    print(f"Test Blocks:        {TEST_BLOCKS}")
    print(f"Val Speaker Frac:   {VAL_SPEAKER_FRACTION}")
    print(f"Random Seed:        {RANDOM_SEED}")
    print(f"Batch Size:         {BATCH_SIZE}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
