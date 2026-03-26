# CIFAR-10 Image Classification: CNN & Transfer Learning

A deep learning project comparing a custom CNN against MobileNetV2 and ResNet50 transfer learning models on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset (60,000 32×32 color images across 10 classes).

## Installation

Requires Python 3.11. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Install the local package (required for notebook imports)
uv pip install -e .
```

## Project Structure

```
├── src/ih-cnn/
│   ├── models.py      # CNN_CIFAR_10, MVV2_CIFAR_10, RN50_CIFAR_10 classes
│   └── utils.py       # helper functions
├── main.ipynb         # Full pipeline: EDA → training → evaluation → comparison
├── data/              # CIFAR-10 pickle files
├── models/            # Trained keras models and their training history
└── pyproject.toml
```

## Models

| Model | Type | Input | Params (approx.) |
|---|---|---|---|
| `CNN_CIFAR_10` | Custom CNN (10 trainable layers) | 32×32 | ~4.6M |
| `MVV2_CIFAR_10` | MobileNetV2 + head (frozen, upscaled) | 32×32 | ~2.3M |
| `RN50_CIFAR_10` | ResNet50 + head (frozen, upscaled) | 32→96×96 | ~23.6M |

All models share a common interface:

```python
model.callbacks                 # ReduceLROnPlateau and EarlyStopping shared settings
model.history                   # Saved history of the model (if loading a trained model) 
model.train(X_train, y_train, ...)
model.evaluate(X_test, y_test)  # prints accuracy, precision, recall, F1 + confusion matrix
model.save_data("model_name")   # save the model and training history to disk
model.summary()                 # returns the model architecture summary
```

## Dataset

CIFAR-10 consists of 50,000 training and 10,000 test images across 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

After downloading the dataset, unzip the file in the folder `data/`.

## Models

In order to access the ResNet50 model, you need to navigate to the `models/` folder and unzip it.

## Hardware

Optimized for Apple Silicon (M-series) via `tensorflow-metal`. Mixed precision (float16) is enabled for faster GPU throughput. 
