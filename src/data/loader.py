"""
Dysarthria AI — Dataset Loader

Functions for discovering UASpeech WAV files, parsing filenames,
creating metadata DataFrames, assigning labels, and constructing
speaker-independent train/val/test splits.
"""

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.config import (
    UASPEECH_AUDIO_DIR,
    LABEL_CONTROL,
    LABEL_DYSARTHRIC,
    LABEL_MAP,
    TRAIN_BLOCKS,
    TEST_BLOCKS,
    VAL_SPEAKER_FRACTION,
    RANDOM_SEED,
    METADATA_CSV,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
)


# ============================================================
# Filename Pattern
# ============================================================

# Expected format: <Speaker>_<Block>_<Word>_<Microphone>.wav
# Examples: CF02_B1_UW20_M7.wav, M04_B3_CW56_M3.wav
FILENAME_PATTERN = re.compile(
    r"^(?P<speaker>[A-Z]+\d+)_(?P<block>B\d+)_(?P<word>[A-Za-z0-9]+)_(?P<microphone>M\d+)\.wav$"
)


# ============================================================
# Dataset Discovery
# ============================================================

def discover_wav_files(root_dir: Optional[Path] = None) -> list[Path]:
    """
    Recursively find all .wav files under the given directory.

    Args:
        root_dir: Root directory to search. Defaults to UASPEECH_AUDIO_DIR.

    Returns:
        Sorted list of Path objects pointing to .wav files.

    Raises:
        FileNotFoundError: If root_dir does not exist.
    """
    if root_dir is None:
        root_dir = UASPEECH_AUDIO_DIR

    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

    wav_files = sorted(root_dir.rglob("*.wav"))
    print(f"Found {len(wav_files)} WAV files under {root_dir}")
    return wav_files


# ============================================================
# Filename Parsing
# ============================================================

def parse_filename(filename: str) -> Optional[dict]:
    """
    Parse a UASpeech filename into its components.

    Expected format: <Speaker>_<Block>_<Word>_<Microphone>.wav

    Args:
        filename: The filename (not full path) to parse.

    Returns:
        Dictionary with keys: speaker, block, word, microphone.
        Returns None if the filename does not match the expected pattern.
    """
    match = FILENAME_PATTERN.match(filename)
    if match is None:
        return None
    return match.groupdict()


# ============================================================
# Speaker Label Assignment
# ============================================================

def get_speaker_label(speaker_id: str) -> int:
    """
    Determine binary label from speaker ID based on UASpeech convention.

    UASpeech convention:
        - Speaker IDs starting with 'C' (e.g., CF02, CM04) → Control
        - Speaker IDs without 'C' prefix (e.g., F02, M04) → Dysarthric

    Args:
        speaker_id: The speaker identifier string.

    Returns:
        LABEL_CONTROL (0) for control speakers,
        LABEL_DYSARTHRIC (1) for dysarthric speakers.
    """
    if speaker_id.startswith("C"):
        return LABEL_CONTROL
    return LABEL_DYSARTHRIC


def get_label_name(label: int) -> str:
    """Convert numeric label to human-readable name."""
    return LABEL_MAP.get(label, "Unknown")


# ============================================================
# Metadata DataFrame Creation
# ============================================================

def create_metadata_dataframe(wav_paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a metadata DataFrame from a list of WAV file paths.

    Each file is parsed for speaker, block, word, and microphone.
    Files that don't match the expected naming pattern are tracked separately.

    Args:
        wav_paths: List of Path objects pointing to .wav files.

    Returns:
        Tuple of (valid_metadata_df, malformed_df).
        valid_metadata_df columns: filepath, filename, speaker, block, word,
                                   microphone, label, label_name
        malformed_df columns: filepath, filename
    """
    valid_records = []
    malformed_records = []

    for path in wav_paths:
        filename = path.name
        parsed = parse_filename(filename)

        if parsed is None:
            malformed_records.append({
                "filepath": str(path),
                "filename": filename,
            })
        else:
            label = get_speaker_label(parsed["speaker"])
            valid_records.append({
                "filepath": str(path),
                "filename": filename,
                "speaker": parsed["speaker"],
                "block": parsed["block"],
                "word": parsed["word"],
                "microphone": parsed["microphone"],
                "label": label,
                "label_name": get_label_name(label),
            })

    valid_df = pd.DataFrame(valid_records)
    malformed_df = pd.DataFrame(malformed_records)

    print(f"Valid filenames:     {len(valid_df)}")
    print(f"Malformed filenames: {len(malformed_df)}")

    return valid_df, malformed_df


# ============================================================
# Data Splitting (Speaker-Independent)
# ============================================================

def create_splits(
    metadata_df: pd.DataFrame,
    train_blocks: Optional[list[str]] = None,
    test_blocks: Optional[list[str]] = None,
    val_speaker_fraction: float = VAL_SPEAKER_FRACTION,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create speaker-independent train/val/test splits.

    Strategy:
        - Test set: all files from TEST_BLOCKS (B2)
        - Train + Val: files from TRAIN_BLOCKS (B1, B3)
        - Validation: a fraction of training SPEAKERS are held out entirely
          (all their files go to validation) to prevent speaker leakage.

    The validation speakers are selected proportionally from each class
    to maintain class balance.

    Args:
        metadata_df: DataFrame with columns: speaker, block, label, etc.
        train_blocks: List of block IDs for training (default from config).
        test_blocks: List of block IDs for testing (default from config).
        val_speaker_fraction: Fraction of training speakers for validation.
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    if train_blocks is None:
        train_blocks = TRAIN_BLOCKS
    if test_blocks is None:
        test_blocks = TEST_BLOCKS

    rng = np.random.RandomState(random_seed)

    # Step 1: Split by block
    test_df = metadata_df[metadata_df["block"].isin(test_blocks)].copy()
    trainval_df = metadata_df[metadata_df["block"].isin(train_blocks)].copy()

    # Step 2: Get unique speakers in train+val, grouped by class
    trainval_speakers = trainval_df[["speaker", "label"]].drop_duplicates()

    val_speakers = []
    for label_val in trainval_speakers["label"].unique():
        class_speakers = trainval_speakers[trainval_speakers["label"] == label_val]["speaker"].values
        n_val = max(1, int(len(class_speakers) * val_speaker_fraction))
        selected = rng.choice(class_speakers, size=n_val, replace=False)
        val_speakers.extend(selected)

    val_speakers = set(val_speakers)

    # Step 3: Split train/val by speaker
    val_df = trainval_df[trainval_df["speaker"].isin(val_speakers)].copy()
    train_df = trainval_df[~trainval_df["speaker"].isin(val_speakers)].copy()

    # Print summary
    print("=" * 60)
    print("Data Split Summary (Speaker-Independent)")
    print("=" * 60)
    print(f"Train:  {len(train_df):>6} samples | "
          f"{train_df['speaker'].nunique():>2} speakers | "
          f"Blocks: {sorted(train_df['block'].unique())}")
    print(f"Val:    {len(val_df):>6} samples | "
          f"{val_df['speaker'].nunique():>2} speakers | "
          f"Blocks: {sorted(val_df['block'].unique())}")
    print(f"Test:   {len(test_df):>6} samples | "
          f"{test_df['speaker'].nunique():>2} speakers | "
          f"Blocks: {sorted(test_df['block'].unique())}")
    print()

    # Verify no speaker leakage
    train_speakers = set(train_df["speaker"].unique())
    val_speakers_actual = set(val_df["speaker"].unique())
    test_speakers = set(test_df["speaker"].unique())

    train_val_overlap = train_speakers & val_speakers_actual
    if train_val_overlap:
        raise ValueError(f"Speaker leakage between train and val: {train_val_overlap}")

    print("✓ No speaker leakage between train and val")
    print(f"  Train speakers: {sorted(train_speakers)}")
    print(f"  Val speakers:   {sorted(val_speakers_actual)}")
    print()

    # Class distribution per split
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        counts = df["label_name"].value_counts()
        print(f"  {name} class distribution:")
        for cls, cnt in counts.items():
            print(f"    {cls}: {cnt}")
        print()

    return train_df, val_df, test_df


def save_split_metadata(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Save train/val/test DataFrames as CSV files."""
    for df, path, name in [
        (train_df, TRAIN_CSV, "Train"),
        (val_df, VAL_CSV, "Val"),
        (test_df, TEST_CSV, "Test"),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Saved {name} metadata: {path} ({len(df)} rows)")


# ============================================================
# Metadata I/O
# ============================================================

def save_metadata(df: pd.DataFrame, path: Optional[Path] = None) -> None:
    """Save metadata DataFrame to CSV."""
    if path is None:
        path = METADATA_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved metadata to {path} ({len(df)} rows)")


def load_metadata(path: Optional[Path] = None) -> pd.DataFrame:
    """Load metadata DataFrame from CSV."""
    if path is None:
        path = METADATA_CSV
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    df = pd.read_csv(path)
    print(f"Loaded metadata from {path} ({len(df)} rows)")
    return df
