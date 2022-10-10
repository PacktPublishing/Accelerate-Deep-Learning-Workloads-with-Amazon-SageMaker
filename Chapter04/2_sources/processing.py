import tensorflow as tf
import tensorflow.keras as keras
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse
import numpy as np
import pandas as pd

WIDTH = 224
HEIGHT = 224


def _get_class_lookup(lookup_file):
    return pd.read_csv(os.path.join(lookup_file, "class_dict.csv"))


def main(args):

    class_lookup = _get_class_lookup(args.lookup_location)

    dataset = keras.utils.image_dataset_from_directory(
        args.data_location,
        labels="inferred",
        label_mode="int",
        class_names=None,
        color_mode="rgb",
        batch_size=args.batch_size,
        image_size=(WIDTH, HEIGHT),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False,
    )

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    max_augmentations = args.max_augmentations
    augmented_root_dir = args.output_location

    i = 0

    print("Start image augmentation")

    for batch_data, batch_labels in dataset.as_numpy_iterator():
        print(f"Processing batch with index {i} out from {len(dataset)}")

        for image, label in zip(batch_data, batch_labels):

            label_name = class_lookup.iloc[label]["class"]
            image_save_dir = os.path.join(augmented_root_dir, label_name)
            os.makedirs(image_save_dir, exist_ok=True)

            j = 0
            image = np.expand_dims(image, axis=0)

            # generate 5 new augmented images
            for batch in datagen.flow(
                image,
                batch_size=1,
                save_to_dir=image_save_dir,
                save_prefix="augmented",
                save_format="jpeg",
            ):
                j += 1
                if j > max_augmentations:
                    break
        i += 1

        if args.max_samples is not None:
            if i > args.max_samples:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_augmentations", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--data_location", type=str, default=None)
    parser.add_argument("--lookup_location", type=str, default=None)
    parser.add_argument("--output_location", type=str, default=None)
    args, remainder = parser.parse_known_args()

    print(args)
    print(os.environ)

    main(args)
