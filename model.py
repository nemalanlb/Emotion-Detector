"""
Module 3 (part 1): Model Definition
--------------------------------------
Defines the CNN architecture described in the project report:

Input (48x48x1) -> Conv2D -> ReLU -> MaxPooling -> Conv2D -> ReLU ->
MaxPooling -> Flatten -> Dense -> Dropout -> Output (7 classes)
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)

from preprocess import NUM_CLASSES, IMG_SIZE  # noqa: F401 (IMG_SIZE used below)


def build_emotion_model():
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", padding="same",
                   input_shape=(IMG_SIZE, IMG_SIZE, 1)),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            Conv2D(128, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            Flatten(),
            Dense(256, activation="relu"),
            BatchNormalization(),
            Dropout(0.5),

            Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    m = build_emotion_model()
    m.summary()
