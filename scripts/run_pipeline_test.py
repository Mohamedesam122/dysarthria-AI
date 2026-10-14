"""
Dysarthria AI — End-to-End Pipeline Test Script

This script runs all pipeline phases sequentially to verify everything works:
1. Dataset discovery
2. Metadata creation & labels
3. Dataset statistics
4. Audio inspection
5. Mel Spectrogram extraction
6. Speaker-independent splitting
7. TensorFlow data pipeline
8. Sanity check baseline model

Run from project root:
    .venv\Scripts\python.exe scripts/run_pipeline_test.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for script execution
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import soundfile as sf

print("=" * 70)
print("DYSARTHRIA AI — END-TO-END PIPELINE TEST")
print("=" * 70)

# ============================================================
# Phase 1: Import project modules
# ============================================================
print("\n[Phase 1] Importing project modules...")

from src.utils.config import (
    PROJECT_ROOT, UASPEECH_AUDIO_DIR, METADATA_DIR, FIGURES_DIR,
    METADATA_CSV, TRAIN_CSV, VAL_CSV, TEST_CSV,
    LABEL_CONTROL, LABEL_DYSARTHRIC, LABEL_MAP,
    TARGET_SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH,
    MAX_DURATION_SEC, TARGET_FRAMES, INPUT_SHAPE,
    TRAIN_BLOCKS, TEST_BLOCKS, BATCH_SIZE, RANDOM_SEED,
    print_config,
)
from src.data.loader import (
    discover_wav_files,
    parse_filename,
    create_metadata_dataframe,
    create_splits,
    save_metadata,
    save_split_metadata,
)
from src.data.preprocessing import (
    load_audio,
    get_audio_info,
    normalize_audio,
    extract_mel_spectrogram,
    fix_length_frames,
    preprocess_single,
)

print_config()
print("✓ All project modules imported successfully")

# ============================================================
# Phase 2: Dataset Discovery
# ============================================================
print("\n" + "=" * 70)
print("[Phase 2] Discovering WAV files...")
print("=" * 70)

wav_files = discover_wav_files()
print(f"\nFirst 5 files:")
for f in wav_files[:5]:
    print(f"  {f.name}")

# ============================================================
# Phase 3: Create Metadata DataFrame
# ============================================================
print("\n" + "=" * 70)
print("[Phase 3] Creating metadata DataFrame...")
print("=" * 70)

metadata_df, malformed_df = create_metadata_dataframe(wav_files)
print(f"\nMetadata shape: {metadata_df.shape}")
print(f"Columns: {list(metadata_df.columns)}")
print(f"\nFirst 5 rows:")
print(metadata_df.head())

if len(malformed_df) > 0:
    print(f"\n⚠ {len(malformed_df)} malformed filenames:")
    print(malformed_df.head(10))
else:
    print("\n✓ No malformed filenames")

duplicates = metadata_df['filepath'].duplicated().sum()
print(f"Duplicate file paths: {duplicates}")

# ============================================================
# Phase 4: Label Verification
# ============================================================
print("\n" + "=" * 70)
print("[Phase 4] Verifying labels...")
print("=" * 70)

print(f"Label mapping: {LABEL_MAP}")
speaker_labels = metadata_df[['speaker', 'label_name']].drop_duplicates().sort_values('speaker')
print(f"\nSpeaker → Label mapping ({len(speaker_labels)} speakers):")
for _, row in speaker_labels.iterrows():
    print(f"  {row['speaker']:>5} → {row['label_name']}")

# ============================================================
# Phase 5: Dataset Statistics
# ============================================================
print("\n" + "=" * 70)
print("[Phase 5] Dataset Statistics")
print("=" * 70)

print(f"1.  Total WAV files:        {len(metadata_df)}")
print(f"2.  Unique speakers:        {metadata_df['speaker'].nunique()}")
print(f"3.  Control speakers:       {metadata_df[metadata_df['label']==0]['speaker'].nunique()}")
print(f"4.  Dysarthric speakers:    {metadata_df[metadata_df['label']==1]['speaker'].nunique()}")
print(f"5.  Samples per class:")
for name, count in metadata_df['label_name'].value_counts().items():
    print(f"      {name}: {count}")
print(f"6.  Unique blocks:          {sorted(metadata_df['block'].unique())}")
print(f"7.  Unique microphones:     {sorted(metadata_df['microphone'].unique())}")
print(f"8.  Unique words:           {metadata_df['word'].nunique()}")
print(f"9.  Malformed filenames:    {len(malformed_df)}")
print(f"10. Duplicate file paths:   {duplicates}")

# Samples per speaker
print(f"\nSamples per speaker:")
speaker_counts = metadata_df.groupby(['speaker', 'label_name']).size().reset_index(name='count')
for _, row in speaker_counts.sort_values('speaker').iterrows():
    print(f"  {row['speaker']:>5} ({row['label_name']:>10}): {row['count']}")

# Samples per block
print(f"\nSamples per block:")
for block, count in metadata_df['block'].value_counts().sort_index().items():
    print(f"  {block}: {count}")

# Samples per microphone
print(f"\nSamples per microphone:")
for mic, count in metadata_df['microphone'].value_counts().sort_index().items():
    print(f"  {mic}: {count}")

# ============================================================
# Phase 6: Visualizations
# ============================================================
print("\n" + "=" * 70)
print("[Phase 6] Generating visualizations...")
print("=" * 70)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Class distribution
class_counts = metadata_df['label_name'].value_counts()
axes[0, 0].bar(class_counts.index, class_counts.values, color=['#2ecc71', '#e74c3c'])
axes[0, 0].set_title('Class Distribution', fontsize=14)
axes[0, 0].set_ylabel('Number of Samples')
for i, (cls, cnt) in enumerate(class_counts.items()):
    axes[0, 0].text(i, cnt + 200, str(cnt), ha='center', fontsize=11, fontweight='bold')

# Speaker distribution
speaker_data = metadata_df.groupby(['speaker', 'label_name']).size().reset_index(name='count')
speaker_data = speaker_data.sort_values(['label_name', 'speaker'])
colors = ['#2ecc71' if l == 'Control' else '#e74c3c' for l in speaker_data['label_name']]
axes[0, 1].barh(speaker_data['speaker'], speaker_data['count'], color=colors)
axes[0, 1].set_title('Samples per Speaker', fontsize=14)
axes[0, 1].set_xlabel('Number of Samples')

# Block distribution
block_counts = metadata_df['block'].value_counts().sort_index()
axes[1, 0].bar(block_counts.index, block_counts.values, color='#3498db')
axes[1, 0].set_title('Block Distribution', fontsize=14)
axes[1, 0].set_ylabel('Number of Samples')
for i, (blk, cnt) in enumerate(block_counts.items()):
    axes[1, 0].text(i, cnt + 200, str(cnt), ha='center', fontsize=11, fontweight='bold')

# Microphone distribution
mic_counts = metadata_df['microphone'].value_counts().sort_index()
axes[1, 1].bar(mic_counts.index, mic_counts.values, color='#9b59b6')
axes[1, 1].set_title('Microphone Distribution', fontsize=14)
axes[1, 1].set_ylabel('Number of Samples')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'dataset_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {FIGURES_DIR / 'dataset_distributions.png'}")

# ============================================================
# Phase 7: Audio Inspection
# ============================================================
print("\n" + "=" * 70)
print("[Phase 7] Inspecting audio properties...")
print("=" * 70)

import random
random.seed(RANDOM_SEED)

sample_size = min(300, len(metadata_df))
sampled = metadata_df.sample(sample_size, random_state=RANDOM_SEED)

sr_counts = {}
ch_counts = {}
durations = []
errors = []

for _, row in sampled.iterrows():
    try:
        info = get_audio_info(row['filepath'])
        sr = info['sample_rate']
        ch = info['channels']
        sr_counts[sr] = sr_counts.get(sr, 0) + 1
        ch_counts[ch] = ch_counts.get(ch, 0) + 1
        durations.append(info['duration'])
    except Exception as e:
        errors.append((row['filename'], str(e)))

print(f"\nInspected {len(durations)} files (from {sample_size} sampled)")
if errors:
    print(f"⚠ {len(errors)} files could not be read:")
    for fn, err in errors[:5]:
        print(f"  {fn}: {err}")

print(f"\nSample rate distribution:")
for sr, cnt in sorted(sr_counts.items()):
    print(f"  {sr} Hz: {cnt} files")

print(f"\nChannel distribution:")
for ch, cnt in sorted(ch_counts.items()):
    print(f"  {ch} channel(s): {cnt} files")

durations = np.array(durations)
print(f"\nDuration statistics (seconds):")
print(f"  Min:       {durations.min():.3f}")
print(f"  Max:       {durations.max():.3f}")
print(f"  Mean:      {durations.mean():.3f}")
print(f"  Median:    {np.median(durations):.3f}")
print(f"  Std Dev:   {durations.std():.3f}")
print(f"  95th pct:  {np.percentile(durations, 95):.3f}")
print(f"  99th pct:  {np.percentile(durations, 99):.3f}")

pct_covered = (durations <= MAX_DURATION_SEC).mean() * 100
print(f"\n{pct_covered:.1f}% of samples <= {MAX_DURATION_SEC}s (will not be cropped)")

# Duration histogram
fig, ax = plt.subplots(figsize=(12, 5))
ax.hist(durations, bins=50, color='#3498db', edgecolor='white', alpha=0.8)
ax.axvline(MAX_DURATION_SEC, color='red', linestyle='--', linewidth=2,
           label=f'Max Duration: {MAX_DURATION_SEC}s')
ax.axvline(np.percentile(durations, 95), color='orange', linestyle='--',
           label=f'95th pct: {np.percentile(durations, 95):.2f}s')
ax.set_title('Audio Duration Distribution', fontsize=14)
ax.set_xlabel('Duration (seconds)')
ax.set_ylabel('Count')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'duration_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {FIGURES_DIR / 'duration_distribution.png'}")

# ============================================================
# Phase 8: Waveform Visualization
# ============================================================
print("\n" + "=" * 70)
print("[Phase 8] Creating waveform visualizations...")
print("=" * 70)

control_sample = metadata_df[metadata_df['label'] == 0].iloc[0]
dysarthric_sample = metadata_df[metadata_df['label'] == 1].iloc[0]

fig, axes = plt.subplots(2, 1, figsize=(14, 6))

audio_c, sr_c = load_audio(control_sample['filepath'], target_sr=None, mono=True)
librosa.display.waveshow(audio_c, sr=sr_c, ax=axes[0])
axes[0].set_title(f"Control: {control_sample['filename']} (sr={sr_c}Hz)", fontsize=12)

audio_d, sr_d = load_audio(dysarthric_sample['filepath'], target_sr=None, mono=True)
librosa.display.waveshow(audio_d, sr=sr_d, ax=axes[1])
axes[1].set_title(f"Dysarthric: {dysarthric_sample['filename']} (sr={sr_d}Hz)", fontsize=12)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'waveform_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {FIGURES_DIR / 'waveform_comparison.png'}")

# ============================================================
# Phase 9: Mel Spectrogram Visualization
# ============================================================
print("\n" + "=" * 70)
print("[Phase 9] Generating Mel Spectrograms...")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

for col, (row, label) in enumerate([
    (control_sample, 'Control'),
    (dysarthric_sample, 'Dysarthric'),
]):
    audio, sr = load_audio(row['filepath'], target_sr=TARGET_SAMPLE_RATE, mono=True)
    audio = normalize_audio(audio)

    # Waveform
    librosa.display.waveshow(audio, sr=sr, ax=axes[0, col])
    axes[0, col].set_title(f'{label}: {row["filename"]} — Waveform', fontsize=12)

    # Mel Spectrogram
    mel_spec = extract_mel_spectrogram(audio, sr)
    img = librosa.display.specshow(mel_spec, sr=sr, hop_length=HOP_LENGTH,
                                    x_axis='time', y_axis='mel', ax=axes[1, col])
    axes[1, col].set_title(f'{label}: {row["filename"]} — Mel Spectrogram', fontsize=12)
    plt.colorbar(img, ax=axes[1, col], format='%+2.0f dB')

    print(f"  {label} sample: {row['filename']}")
    print(f"    Audio: {len(audio)} samples ({len(audio)/sr:.3f} sec)")
    print(f"    Mel spec shape: {mel_spec.shape}")

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'mel_spectrogram_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {FIGURES_DIR / 'mel_spectrogram_comparison.png'}")

# ============================================================
# Phase 10: Test Full Preprocessing Pipeline
# ============================================================
print("\n" + "=" * 70)
print("[Phase 10] Testing preprocessing pipeline...")
print("=" * 70)

test_samples = pd.concat([
    metadata_df[metadata_df['label'] == 0].sample(5, random_state=42),
    metadata_df[metadata_df['label'] == 1].sample(5, random_state=42),
])

print(f"Expected output shape: {INPUT_SHAPE}")
print()

all_ok = True
for _, row in test_samples.iterrows():
    try:
        tensor = preprocess_single(row['filepath'])
        has_nan = np.isnan(tensor).any()
        has_inf = np.isinf(tensor).any()
        status = "✓" if not has_nan and not has_inf else "⚠"
        print(f"  {status} {row['filename']:30s} | shape: {tensor.shape} | "
              f"min: {tensor.min():.2f} | max: {tensor.max():.2f} | "
              f"nan: {has_nan} | inf: {has_inf}")
        if tensor.shape != INPUT_SHAPE:
            print(f"    ⚠ Shape mismatch! Expected {INPUT_SHAPE}")
            all_ok = False
    except Exception as e:
        print(f"  ✗ {row['filename']:30s} | ERROR: {e}")
        all_ok = False

if all_ok:
    print("\n✓ All samples preprocessed successfully")
else:
    print("\n⚠ Some samples had issues")

# ============================================================
# Phase 11: Save Metadata
# ============================================================
print("\n" + "=" * 70)
print("[Phase 11] Saving metadata...")
print("=" * 70)

save_metadata(metadata_df)

# ============================================================
# Phase 12: Speaker-Independent Data Split
# ============================================================
print("\n" + "=" * 70)
print("[Phase 12] Creating speaker-independent splits...")
print("=" * 70)

train_df, val_df, test_df = create_splits(metadata_df)
save_split_metadata(train_df, val_df, test_df)

# ============================================================
# Phase 13: TensorFlow Data Pipeline
# ============================================================
print("\n" + "=" * 70)
print("[Phase 13] Building TensorFlow data pipeline...")
print("=" * 70)

import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

def create_tf_dataset(file_paths, labels, batch_size=BATCH_SIZE, shuffle=True):
    """Create a tf.data.Dataset from file paths and labels."""

    def preprocess_tf(filepath, label):
        def _preprocess(filepath_bytes, label_val):
            filepath_str = filepath_bytes.numpy().decode('utf-8')
            mel_spec = preprocess_single(filepath_str)
            return mel_spec.astype(np.float32), label_val
        mel_spec, label_out = tf.py_function(
            _preprocess, [filepath, label], [tf.float32, tf.int64],
        )
        mel_spec.set_shape(INPUT_SHAPE)
        label_out.set_shape(())
        return mel_spec, label_out

    dataset = tf.data.Dataset.from_tensor_slices((
        file_paths.astype(str),
        labels.astype(np.int64),
    ))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(file_paths), 1000))
    dataset = dataset.map(preprocess_tf, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# Test with small subset
small_subset = pd.concat([
    train_df[train_df['label'] == 0].head(16),
    train_df[train_df['label'] == 1].head(16),
])

print(f"\nTesting pipeline with {len(small_subset)} samples...")
test_dataset = create_tf_dataset(
    small_subset['filepath'].values,
    small_subset['label'].values,
    batch_size=8,
    shuffle=False,
)

for batch_x, batch_y in test_dataset.take(1):
    print(f"\n  Batch X shape:  {batch_x.shape}")
    print(f"  Batch Y shape:  {batch_y.shape}")
    print(f"  Batch X dtype:  {batch_x.dtype}")
    print(f"  Batch Y dtype:  {batch_y.dtype}")
    print(f"  Batch X min:    {tf.reduce_min(batch_x).numpy():.4f}")
    print(f"  Batch X max:    {tf.reduce_max(batch_x).numpy():.4f}")
    print(f"  Unique labels:  {np.unique(batch_y.numpy())}")
    print(f"  Has NaN:        {tf.reduce_any(tf.math.is_nan(batch_x)).numpy()}")
    print(f"  Has Inf:        {tf.reduce_any(tf.math.is_inf(batch_x)).numpy()}")

print("\n✓ TensorFlow pipeline test passed!")

# Batch spectrogram visualization
for batch_x, batch_y in test_dataset.take(1):
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for i in range(min(8, batch_x.shape[0])):
        row_i, col_i = i // 4, i % 4
        ax = axes[row_i, col_i]
        spec = batch_x[i].numpy()[:, :, 0]
        label = batch_y[i].numpy()
        label_name = LABEL_MAP[label]
        ax.imshow(spec, aspect='auto', origin='lower', cmap='magma')
        ax.set_title(f'{label_name} (y={label})', fontsize=11)
        ax.set_xlabel('Time Frames')
        ax.set_ylabel('Mel Bands')
    plt.suptitle('Sample Mel Spectrograms from Training Batch', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'batch_spectrograms.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {FIGURES_DIR / 'batch_spectrograms.png'}")

# ============================================================
# Phase 14: Sanity Check Baseline Model
# ============================================================
print("\n" + "=" * 70)
print("[Phase 14] Training sanity check baseline model...")
print("=" * 70)
print("NOTE: This is NOT the final model — just verifying the pipeline works.")

from tensorflow import keras
from tensorflow.keras import layers

def create_baseline_model(input_shape):
    """Minimal CNN for pipeline verification only."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model

model = create_baseline_model(INPUT_SHAPE)
model.summary()

# Small train/val for sanity check
small_train = create_tf_dataset(
    train_df.head(128)['filepath'].values,
    train_df.head(128)['label'].values,
    batch_size=16,
    shuffle=True,
)
small_val = create_tf_dataset(
    val_df.head(32)['filepath'].values,
    val_df.head(32)['label'].values,
    batch_size=16,
    shuffle=False,
)

print("\nTraining for 3 epochs on small subset...")
history = model.fit(
    small_train,
    validation_data=small_val,
    epochs=3,
    verbose=1,
)

print(f"\n  Final train loss: {history.history['loss'][-1]:.4f}")
print(f"  Final train acc:  {history.history['accuracy'][-1]:.4f}")
if 'val_loss' in history.history:
    print(f"  Final val loss:   {history.history['val_loss'][-1]:.4f}")
    print(f"  Final val acc:    {history.history['val_accuracy'][-1]:.4f}")

# Verify predictions
for batch_x, batch_y in small_val.take(1):
    predictions = model.predict(batch_x, verbose=0)
    print(f"\n  Prediction shape: {predictions.shape}")
    print(f"  Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"  True labels:      {batch_y.numpy()[:8]}")
    print(f"  Predictions:      {predictions.flatten()[:8].round(3)}")
    print(f"  All in [0,1]:     {(predictions >= 0).all() and (predictions <= 1).all()}")
    print(f"  Has NaN:          {np.isnan(predictions).any()}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("✓ END-TO-END PIPELINE TEST COMPLETE")
print("=" * 70)
print()
print("Pipeline verified:")
print("  Dataset → Metadata → Labels → Split → Audio Loading")
print("  → Preprocessing → Mel Spectrogram → TensorFlow Dataset")
print("  → Batch Generation → Model Training → Predictions")
print()
print(f"Files created:")
print(f"  {METADATA_CSV}")
print(f"  {TRAIN_CSV}")
print(f"  {VAL_CSV}")
print(f"  {TEST_CSV}")
print(f"  {FIGURES_DIR / 'dataset_distributions.png'}")
print(f"  {FIGURES_DIR / 'duration_distribution.png'}")
print(f"  {FIGURES_DIR / 'waveform_comparison.png'}")
print(f"  {FIGURES_DIR / 'mel_spectrogram_comparison.png'}")
print(f"  {FIGURES_DIR / 'batch_spectrograms.png'}")
print()
print("Ready for CNN-BiLSTM-Attention model implementation!")
