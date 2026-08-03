"""
Module 2: Image Preprocessing (folder-based dataset version)
------------------------------------------------------------------
This version matches the Kaggle "msambare/fer2013" dataset layout:

    emotion_detector/
        train/
            angry/    *.jpg
            disgust/  *.jpg
            fear/     *.jpg
            happy/    *.jpg
            neutral/  *.jpg
            sad/      *.jpg
            surprise/ *.jpg
        test/
            (same subfolder structure)

Keras' ImageDataGenerator.flow_from_directory() reads images directly
off disk, resizes them, converts to grayscale, and normalizes pixel
values -- so we don't need to manually parse a CSV.

IMPORTANT: flow_from_directory assigns class indices by sorting folder
names alphabetically:
    0 angry, 1 disgust, 2 fear, 3 happy, 4 neutral, 5 sad, 6 surprise
realtime_detection.py's EMOTIONS list is ordered to match this exactly.
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 48
NUM_CLASSES = 7
BATCH_SIZE = 64

TRAIN_DIR = "train"
TEST_DIR = "test"

# Alphabetical order == flow_from_directory's default class index order
EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise",
}


def get_data_generators(train_dir=TRAIN_DIR, test_dir=TEST_DIR,
                         batch_size=BATCH_SIZE, validation_split=0.1):
    """
    Returns (train_generator, val_generator, test_generator).

    A slice of the train/ folder is held out as validation data via
    validation_split, and test/ is used as the final held-out test set.
    """

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=validation_split,
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=True,
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    return train_generator, val_generator, test_generator


if __name__ == "__main__":
    train_gen, val_gen, test_gen = get_data_generators()
    print("Class indices:", train_gen.class_indices)
    print("Train samples:", train_gen.samples)
    print("Val samples:", val_gen.samples)
    print("Test samples:", test_gen.samples)
