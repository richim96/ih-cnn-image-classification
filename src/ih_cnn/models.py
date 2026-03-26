"""Models module."""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from abc import ABC
from uuid import uuid4
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers, applications as app
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
]


class _BaseModel(ABC):
    """Shared interface wrapping a Keras model with .fit(), ad a custom .evaluate()."""

    def __init__(self, model_path: str | None = None):
        """Subclasses must set self.model in their __init__."""
        self.callbacks: list = [
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6), # 0.00001
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ]
        self.history: dict[str, list[float]] = {}

        if model_path:
            self.model = keras.models.load_model(model_path)
            history_path = model_path.replace(".keras", "_history.json")
            self.history = self._load_history(history_path)

    def train(self, *args, **kwargs):
        """Train the model (delegates to the underlying model)."""
        return self.model.fit(*args, **kwargs)
    
    def save_data(self, model_family: str):
        """Save the model to disk (delegates to the underlying model)."""
        model_path = "models/" + model_family + f"/{uuid4()}.keras"
        history_path = model_path.replace(".keras", "_history.json")

        with open(history_path, "w") as f:
            json.dump(self.history.history, f)

        self.model.save(model_path)

        print(f"Model data saved.")

    def evaluate(self, X, y):
        """Print accuracy, precision, recall, F1-score and plot confusion matrix."""
        y_pred = np.argmax(self.model.predict(X), axis=1)
        y_true = y if y.ndim == 1 else np.argmax(y, axis=1)

        print(classification_report(y_true, y_pred, target_names=CLASSES))

        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=CLASSES,
            yticklabels=CLASSES,
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.show()

    def summary(self):
        """Print the model architecture (delegates to the underlying model)."""
        return self.model.summary()
    
    def _compile_adam_classification(self, metrics: list[str]=[]):
        """Apply a standard compilation for a classification problem with Adam
        optimizer (delegates to the underlying model).
        """
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=3e-4), # 0.0003
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"] + metrics,
        )

    def _load_history(self, path: str) -> dict[str, list[float]]:
        """Load training history from a JSON file."""
        with open(path) as f:
            return json.load(f)


class CNN_CIFAR_10(_BaseModel):
    """Custom sequential CNN for CIFAR-10 (8 layers)."""

    def __init__(self, model_path: str | None = None):
        super().__init__(model_path)

        if not model_path:
            self.model = keras.Sequential([
                layers.Input(shape=(32, 32, 3)),

                # Data augmentation
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.05),
                layers.RandomZoom(0.05),
                layers.RandomContrast(0.05),

                # Conv layers
                layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),

                layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),

                layers.Conv2D(128, (3,3), padding="same", activation="relu"),
                layers.BatchNormalization(),

                layers.Conv2D(256, (3,3), padding="same", activation="relu"),
                layers.BatchNormalization(),

                # Classification head
                layers.Flatten(),
                layers.Dropout(0.4),
                layers.Dense(256, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(10, activation="softmax", dtype="float32"),
            ])
            
            self._compile_adam_classification()


class MNV2_CIFAR_10(_BaseModel):
    """MobileNetV2 transfer learning model for CIFAR-10."""

    def __init__(self, model_path: str | None = None):
        super().__init__(model_path)

        if not model_path:
            mnv2 = app.MobileNetV2(include_top=False, weights="imagenet", pooling="avg")
            mnv2.trainable = False

            inputs = keras.Input(shape=(32, 32, 3))
            x = layers.Resizing(96, 96)(inputs)
            x = layers.Rescaling(scale=2.0, offset=-1.0)(x)  # [0,1] → [-1,1]
            x = mnv2(x, training=False)
            x = layers.Dense(256, activation="relu")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            outputs = layers.Dense(10, activation="softmax", dtype="float32")(x)

            self.model = keras.Model(inputs, outputs)
            self._compile_adam_classification()


class RN50_CIFAR_10(_BaseModel):
    """ResNet50 transfer learning model for CIFAR-10."""

    def __init__(self, model_path: str | None = None):
        super().__init__(model_path)

        if not model_path:
            rn50 = app.ResNet50(include_top=False, weights="imagenet", pooling="avg")
            rn50.trainable = False

            inputs = keras.Input(shape=(32, 32, 3))
            x = layers.Resizing(96, 96)(inputs)
            x = layers.Rescaling(scale=255.0)(x)        # [0,1] → [0,255]
            x = app.resnet50.preprocess_input(x)        # mean subtraction + BGR
            x = rn50(x, training=False)
            x = layers.Dense(256, activation="relu")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            outputs = layers.Dense(10, activation="softmax", dtype="float32")(x)

            self.model = keras.Model(inputs, outputs)
            self._compile_adam_classification()
