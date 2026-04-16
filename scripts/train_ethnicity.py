"""Train the ethnicity CNN on the UTKFace Kaggle dataset.

Usage
-----
    # Install training-only deps first:
    pip install tensorflow pandas "kagglehub[pandas-datasets]"

    # Train. The dataset auto-downloads via kagglehub on first run.
    python scripts/train_ethnicity.py

    # Or point at a local CSV:
    python scripts/train_ethnicity.py --csv /path/to/age_gender.csv

The dataset - `nipunarora8/age-gender-and-ethnicity-face-data-csv` on Kaggle -
contains 48x48 grayscale face crops serialized as a space-separated string of
integers in the ``pixels`` column, along with ``age``, ``gender`` and
``ethnicity`` columns. Ethnicity encoding matches UTKFace:

    0 = White, 1 = Black, 2 = East Asian, 3 = Indian, 4 = Other

The trained model is saved to ``models/ethnicity_cnn.keras`` so the Flask app
picks it up automatically on the next reload.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_OUT = ROOT / "models" / "ethnicity_cnn.keras"
NUM_CLASSES = 5
IMAGE_SIZE = 48


def load_dataframe(csv: str | None):
    """Load the UTKFace CSV either from disk or from Kaggle Hub."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "pandas is required for training. "
            "Install with: pip install pandas"
        ) from exc

    if csv:
        print(f"[train] Loading CSV from {csv}")
        return pd.read_csv(csv)

    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
    except ImportError as exc:
        raise SystemExit(
            "kagglehub is required to auto-download the dataset. "
            "Install with: pip install 'kagglehub[pandas-datasets]'"
        ) from exc

    print("[train] Downloading dataset from Kaggle Hub...")
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "nipunarora8/age-gender-and-ethnicity-face-data-csv",
        "",  # file_path empty - load the primary CSV
    )
    return df


def df_to_arrays(df):
    """Convert the ``pixels`` string column to an (N, 48, 48, 1) array."""
    print(f"[train] First 5 records:\n{df.head()}")
    print(f"[train] Dataset size: {len(df)} rows")

    pixels = df["pixels"].astype(str).values
    X = np.empty((len(pixels), IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32)
    for i, row in enumerate(pixels):
        arr = np.fromstring(row, sep=" ", dtype=np.float32)
        if arr.size != IMAGE_SIZE * IMAGE_SIZE:
            raise ValueError(
                f"Row {i} has {arr.size} pixels (expected {IMAGE_SIZE ** 2})"
            )
        X[i, :, :, 0] = arr.reshape(IMAGE_SIZE, IMAGE_SIZE)
    X /= 255.0

    y = df["ethnicity"].astype(int).values
    return X, y


def build_model():
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=str, default=None,
                    help="Path to the UTKFace CSV (defaults to Kaggle Hub)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        from tensorflow import keras
    except ImportError:
        raise SystemExit(
            "tensorflow is required for training. "
            "Install with: pip install tensorflow"
        )
    from sklearn.model_selection import train_test_split

    df = load_dataframe(args.csv)
    X, y = df_to_arrays(df)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=args.val_split, stratify=y, random_state=args.seed,
    )
    print(f"[train] Train: {X_tr.shape}  Val: {X_val.shape}")

    # Class weights to mitigate imbalance (Indian + Other are smaller classes).
    from collections import Counter
    counts = Counter(y_tr.tolist())
    total = sum(counts.values())
    class_weight = {c: total / (NUM_CLASSES * counts[c]) for c in counts}
    print(f"[train] Class weights: {class_weight}")

    # Simple on-the-fly augmentation.
    augment = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.06),
        keras.layers.RandomZoom(0.08),
        keras.layers.RandomTranslation(0.05, 0.05),
    ], name="augmentation")

    model = build_model()
    model.summary()

    # Wrap augmentation into the training pipeline.
    augmented_input = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    x = augment(augmented_input)
    out = model(x)
    train_model = keras.Model(augmented_input, out)
    train_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, min_lr=1e-5),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
    ]

    train_model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_OUT)
    print(f"[train] Saved model to {MODEL_OUT}")

    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"[train] Final validation: loss={loss:.4f}  acc={acc:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
