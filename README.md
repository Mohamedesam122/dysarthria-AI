# Dysarthria AI

An AI-based speech analysis system designed to detect dysarthric speech using the UASpeech corpus.

## Project Overview

Dysarthria AI builds a reliable end-to-end speech data pipeline for binary classification of speech as **Control** (healthy) or **Dysarthric** (speech disorder associated with cerebral palsy).

### Current Pipeline

```
UASpeech WAV files (noisereduce)
        ↓
Dataset metadata (speaker, block, word, microphone)
        ↓
Speaker/class labels (Control vs Dysarthric)
        ↓
Speaker-independent data split (B1+B3 → Train, B2 → Test)
        ↓
Audio preprocessing (load, mono, resample, normalize)
        ↓
Mel Spectrogram (128 bands, power-to-dB)
        ↓
Fixed-size tensor (pad/crop to uniform shape)
        ↓
TensorFlow data pipeline (tf.data)
        ↓
Sanity check baseline model (small CNN)
```

### Planned Model Architecture

```
Mel Spectrogram
      ↓
    CNN (feature extraction)
      ↓
   BiLSTM (temporal modeling)
      ↓
  Attention (focus on informative regions)
      ↓
  Binary Classification (Control / Dysarthric)
```

> **Note:** The final CNN-BiLSTM-Attention model has NOT been implemented yet. The current focus is on building a correct and reproducible data pipeline.

## Dataset

**UASpeech** — Universally Accessible Speech corpus

- **Version used:** NOISEREDUCE
- **Speakers:** 28 total (13 Control + 15 Dysarthric)
- **Control speakers** (C prefix): CF02, CF03, CF04, CF05, CM01, CM04, CM05, CM06, CM08, CM09, CM10, CM12, CM13
- **Dysarthric speakers** (no C prefix): F02, F03, F04, F05, M01, M04, M05, M07, M08, M09, M10, M11, M12, M14, M16
- **Blocks:** B1, B2, B3
- **Microphones:** M1–M8
- **Filename format:** `<Speaker>_<Block>_<Word>_<Microphone>.wav`

### Label Mapping

| Label | Value | Description |
|-------|-------|-------------|
| Control | 0 | Speakers with C prefix (healthy speech) |
| Dysarthric | 1 | Speakers without C prefix (dysarthric speech) |

## Environment Setup

### Prerequisites

- Python 3.11
- Windows 10/11

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd "speech disorder gp"

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import numpy, pandas, librosa, soundfile, scipy, sklearn, tensorflow; print('All imports OK')"
```

## Project Structure

```
speech disorder gp/
│
├── data/
│   ├── raw/                    # Extracted UASpeech dataset (not in Git)
│   │   └── UASpeech/
│   │       ├── audio/noisereduce/  # WAV files by speaker
│   │       ├── doc/                # Documentation, lexicon
│   │       └── mlf/                # Word-level alignments
│   ├── processed/              # Preprocessed data (generated)
│   └── metadata/               # CSV metadata files
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_audio_preprocessing.ipynb
│
├── src/
│   ├── data/
│   │   ├── loader.py           # Dataset discovery, parsing, splitting
│   │   └── preprocessing.py    # Audio preprocessing pipeline
│   ├── features/
│   │   └── audio_features.py   # Visualization and feature utilities
│   └── utils/
│       └── config.py           # Central configuration
│
├── models/                     # Saved model checkpoints (future)
├── outputs/
│   ├── figures/                # Generated plots
│   ├── logs/                   # Training logs
│   └── results/                # Evaluation results
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Data Exploration

Run `notebooks/01_data_exploration.ipynb` to:
- Discover and parse all WAV files
- Generate metadata with speaker/block/word/microphone fields
- Assign binary labels
- Compute dataset statistics
- Visualize class/speaker/block distributions
- Inspect audio properties (sample rate, channels, duration)

## Preprocessing

Run `notebooks/02_audio_preprocessing.ipynb` to:
- Verify audio sampling rates
- Compute duration statistics
- Generate Mel Spectrograms
- Implement fixed-size padding/cropping
- Create speaker-independent train/val/test splits
- Build TensorFlow data pipeline
- Run end-to-end sanity check

## Data Splitting

**Speaker-independent, block-based split:**

| Split | Blocks | Strategy |
|-------|--------|----------|
| Train | B1, B3 | Majority of speakers |
| Validation | B1, B3 | ~15% of speakers held out entirely |
| Test | B2 | All speakers |

> **Important:** Validation is split at the **speaker level**, not the file level, to prevent speaker leakage.

## Status

### ✅ Completed
- Project structure created
- Configuration module
- Dataset loader with filename validation
- Audio preprocessing pipeline
- Feature extraction utilities
- Data exploration notebook
- Audio preprocessing notebook
- Speaker-independent data splitting
- TensorFlow data pipeline
- End-to-end sanity check

### 🔄 Current Baseline
- Small CNN sanity-check model (verifies pipeline correctness only)

### 📋 Future Work
- CNN-BiLSTM-Attention architecture implementation
- Hyperparameter tuning
- Microphone selection experiments
- Compare NOISEREDUCE vs NORMALIZED versions
- Severity-level classification (beyond binary)
- Cross-validation experiments
