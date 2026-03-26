"""Utils module."""

import pickle
import matplotlib.pyplot as plt


def unpickle(file):
    """Load a CIFAR-10 batch file."""
    with open(file, "rb") as f:
        return pickle.load(f, encoding="bytes")


def plot_training_curves(history: dict[str, list[float]], title: str =""):
    """Plot training and validation loss/accuracy curves."""
    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history["loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title(f"{title} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Accuracy
    axes[1].plot(history["accuracy"], label="Train Accuracy")
    axes[1].plot(history["val_accuracy"], label="Val Accuracy")
    axes[1].set_title(f"{title} - Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
