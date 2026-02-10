"""
Download MNIST data and save as numpy arrays + sample PNG images.
Tries keras first, falls back to sklearn's fetch_openml.
"""

import os
import sys
import numpy as np

CHALLENGE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "challenges", "mnist_digit_recognition"
)
SAMPLES_DIR = os.path.join(CHALLENGE_DIR, "samples")


def download_via_keras():
    """Download MNIST using keras."""
    try:
        from tensorflow.keras.datasets import mnist
        print("Using tensorflow.keras.datasets.mnist...")
    except ImportError:
        try:
            from keras.datasets import mnist
            print("Using keras.datasets.mnist...")
        except ImportError:
            return None

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return X_train, y_train, X_test, y_test


def download_via_sklearn():
    """Download MNIST using sklearn's fetch_openml."""
    print("Using sklearn.datasets.fetch_openml('mnist_784')...")
    from sklearn.datasets import fetch_openml

    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.uint8).reshape(-1, 28, 28)
    y = mnist.target.astype(np.uint8)

    # Standard MNIST split: first 60k train, last 10k test
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test


def save_sample_images(X_train, y_train):
    """Save one sample PNG per digit (0-9)."""
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    try:
        from PIL import Image
    except ImportError:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            for digit in range(10):
                idx = np.where(y_train == digit)[0][0]
                img = X_train[idx]
                fig, ax = plt.subplots(1, 1, figsize=(1, 1), dpi=28)
                ax.imshow(img, cmap="gray")
                ax.axis("off")
                fig.savefig(
                    os.path.join(SAMPLES_DIR, f"sample_{digit}.png"),
                    bbox_inches="tight", pad_inches=0
                )
                plt.close(fig)
            print(f"Saved 10 sample images via matplotlib to {SAMPLES_DIR}")
            return
        except ImportError:
            print("WARNING: Neither PIL nor matplotlib available. Skipping sample images.")
            return

    for digit in range(10):
        idx = np.where(y_train == digit)[0][0]
        img = X_train[idx]
        pil_img = Image.fromarray(img, mode="L")
        path = os.path.join(SAMPLES_DIR, f"sample_{digit}.png")
        pil_img.save(path)

    print(f"Saved 10 sample images via PIL to {SAMPLES_DIR}")


def main():
    os.makedirs(CHALLENGE_DIR, exist_ok=True)

    # Try keras first, then sklearn
    data = download_via_keras()
    if data is None:
        data = download_via_sklearn()

    X_train, y_train, X_test, y_test = data

    # Ensure uint8
    X_train = X_train.astype(np.uint8)
    y_train = y_train.astype(np.uint8)
    X_test = X_test.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    # Print shapes
    print(f"X_train: {X_train.shape}, dtype={X_train.dtype}")
    print(f"y_train: {y_train.shape}, dtype={y_train.dtype}")
    print(f"X_test:  {X_test.shape}, dtype={X_test.dtype}")
    print(f"y_test:  {y_test.shape}, dtype={y_test.dtype}")
    print(f"Label range: {y_train.min()}-{y_train.max()}")
    print(f"Pixel range: {X_train.min()}-{X_train.max()}")

    # Save numpy arrays
    np.save(os.path.join(CHALLENGE_DIR, "mnist_train_images.npy"), X_train)
    np.save(os.path.join(CHALLENGE_DIR, "mnist_train_labels.npy"), y_train)
    np.save(os.path.join(CHALLENGE_DIR, "mnist_test_images.npy"), X_test)
    np.save(os.path.join(CHALLENGE_DIR, "mnist_test_labels.npy"), y_test)
    print(f"Saved .npy files to {CHALLENGE_DIR}")

    # Save sample images
    save_sample_images(X_train, y_train)

    # Verify
    print("\nVerification:")
    for fname in ["mnist_train_images.npy", "mnist_train_labels.npy",
                   "mnist_test_images.npy", "mnist_test_labels.npy"]:
        path = os.path.join(CHALLENGE_DIR, fname)
        arr = np.load(path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  {fname}: shape={arr.shape}, dtype={arr.dtype}, size={size_mb:.1f} MB")

    print("\nDone! MNIST data ready.")


if __name__ == "__main__":
    main()
