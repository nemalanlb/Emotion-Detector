"""
Transfer Learning version of the emotion model, using MobileNetV2
-----------------------------------------------------------------------
Instead of training a CNN from scratch, this starts from MobileNetV2
pretrained on ImageNet (it already knows general visual features like
edges, shapes, textures) and adapts it to emotion classification.
This typically beats a from-scratch small CNN on FER-2013, per the
report's "Future Enhancements" section.

Two-stage training:
  Stage 1: freeze MobileNetV2, train only the new classifier head
  Stage 2: unfreeze the top layers of MobileNetV2 and fine-tune with
           a low learning rate

Usage:
    python train_model_transfer.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 96  # MobileNetV2 needs at least ~96x96; 48x48 FER images get upscaled
BATCH_SIZE = 64
NUM_CLASSES = 7
TRAIN_DIR = "train"
TEST_DIR = "test"
MODEL_SAVE_PATH = "emotion_model_transfer.h5"

STAGE1_EPOCHS = 15   # frozen base, train head only
STAGE2_EPOCHS = 30   # fine-tune top layers of the base


def get_generators():
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.1,
    )
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # color_mode="rgb" -- flow_from_directory auto-converts the grayscale
    # FER images to 3-channel RGB, which MobileNetV2 requires
    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), color_mode="rgb",
        batch_size=BATCH_SIZE, class_mode="categorical", subset="training", shuffle=True,
    )
    val_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), color_mode="rgb",
        batch_size=BATCH_SIZE, class_mode="categorical", subset="validation", shuffle=True,
    )
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR, target_size=(IMG_SIZE, IMG_SIZE), color_mode="rgb",
        batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False,
    )
    return train_gen, val_gen, test_gen


def build_model():
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False  # freeze for stage 1

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model, base_model


def plot_history(history, filename="training_history_transfer.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Model Loss")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved training curves to {filename}")


def main():
    print("Setting up data generators (this may take a moment on first run)...")
    train_gen, val_gen, test_gen = get_generators()
    print("Class indices:", train_gen.class_indices)

    print("Computing class weights...")
    class_labels = np.unique(train_gen.classes)
    weights = compute_class_weight("balanced", classes=class_labels, y=train_gen.classes)
    class_weight_dict = dict(zip(class_labels, weights))

    print("Building MobileNetV2-based model...")
    model, base_model = build_model()
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    ]

    print("\n=== Stage 1: training classifier head (base frozen) ===")
    history1 = model.fit(
        train_gen, validation_data=val_gen, epochs=STAGE1_EPOCHS,
        class_weight=class_weight_dict, callbacks=callbacks,
    )

    print("\n=== Stage 2: fine-tuning top layers of MobileNetV2 ===")
    base_model.trainable = True
    # Only unfreeze the last ~30 layers -- keeps early general features intact
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    history2 = model.fit(
        train_gen, validation_data=val_gen, epochs=STAGE2_EPOCHS,
        class_weight=class_weight_dict, callbacks=callbacks,
    )

    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    # Combine both stages' history for one plot
    combined_history = history1.history.copy()
    for k in combined_history:
        combined_history[k] += history2.history[k]

    class CombinedHistory:
        pass
    ch = CombinedHistory()
    ch.history = combined_history
    plot_history(ch)

    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()