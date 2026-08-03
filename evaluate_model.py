"""
Evaluate a saved model's accuracy on the test set.
--------------------------------------------------------
Loads emotion_model.h5 and reports:
  - Overall test accuracy/loss
  - Per-emotion precision/recall/F1 (classification report)
  - A confusion matrix plot (saved as confusion_matrix.png)

Usage:
    python evaluate_model.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

MODEL_PATH = "emotion_model.h5"
TEST_DIR = "test"
IMG_SIZE = 48
BATCH_SIZE = 64


def main():
    print("Loading model...")
    model = load_model(MODEL_PATH)

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    class_names = list(test_gen.class_indices.keys())

    print("\nEvaluating overall accuracy...")
    loss, acc = model.evaluate(test_gen)
    print(f"Overall Test Accuracy: {acc * 100:.2f}%")
    print(f"Overall Test Loss: {loss:.4f}")

    print("\nGenerating predictions for detailed report...")
    test_gen.reset()
    preds = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes

    print("\nPer-emotion performance:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("\nSaved confusion matrix to confusion_matrix.png")


if __name__ == "__main__":
    main()