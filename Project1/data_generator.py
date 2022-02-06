from enum import Enum
import os
import shutil
from typing import List
from skimage.draw import random_shapes
from skimage.io import imsave
import numpy as np
import cv2
import random

import pickle

# One hot encoding of classes
VERTICAL = np.array([1, 0, 0, 0])
HORIZONTAL = np.array([0, 1, 0, 0])
RECTANGLE = np.array([0, 0, 1, 0])
CROSS = np.array([0, 0, 0, 1])


class DataGenerator:
    def __init__(
        self,
        rec_width_range: List[int],
        rec_height_range: List[int],
        ver_length_range: List[int],
        hor_length_range: List[int],
        cro_length_range: List[int],
        nr_images: int,
        image_size: int,
        noise: float,
        train_size: float = 0.70,
        val_size: float = 0.20,
        test_size: float = 0.10,
    ):
        self.rec_width_range = rec_width_range
        self.rec_height_range = rec_height_range
        self.ver_length_range = ver_length_range
        self.hor_length_range = hor_length_range
        self.cro_length_range = cro_length_range

        self.image_size = image_size
        self.nr_images = nr_images
        self.noise = noise

        self.paths = ["train", "val", "test"]
        self.sizes = [
            round(train_size * nr_images),
            round(val_size * nr_images),
            round(test_size * nr_images),
        ]

        self.dataset_file_name = "dataset.p"

    def generate_datasets(self):

        # Generate the datasets
        datasets = {"size: ": self.image_size}  # For being able to convert to 2D again

        for path, size in zip(self.paths, self.sizes):
            data_x = []
            data_y = []

            for _ in range(size // 4):
                rec = np.zeros((self.image_size, self.image_size))
                rec = DataGenerator.generate_noise(
                    DataGenerator.generate_rectangle(
                        rec, self.rec_width_range, self.rec_height_range
                    ),
                    self.noise,
                )
                ver = np.zeros((self.image_size, self.image_size))
                ver = DataGenerator.generate_noise(
                    DataGenerator.generate_vertical_bars(ver, self.ver_length_range),
                    self.noise,
                )
                hor = np.zeros((self.image_size, self.image_size))
                hor = DataGenerator.generate_noise(
                    DataGenerator.generate_horizontal_bars(hor, self.hor_length_range),
                    self.noise,
                )
                cro = np.zeros((self.image_size, self.image_size))
                cro = DataGenerator.generate_noise(
                    DataGenerator.generate_cross(cro, self.cro_length_range), self.noise
                )
                data_x.append(rec.flatten())
                data_x.append(ver.flatten())
                data_x.append(hor.flatten())
                data_x.append(cro.flatten())
                data_y.append(RECTANGLE)
                data_y.append(VERTICAL)
                data_y.append(HORIZONTAL)
                data_y.append(CROSS)

            datasets[f"x_{path}"] = np.array(data_x)
            datasets[f"y_{path}"] = np.array(data_y)

        # Save datasets in a pickle file
        with open("dataset.pickle", "wb") as handle:
            pickle.dump(datasets, handle)

    @staticmethod
    def generate_rectangle(img, width_range: List[int], height_range: List[int]):
        width = random.randint(width_range[0], width_range[1])
        height = random.randint(height_range[0], height_range[1])
        n = img.shape[0]

        corner = (
            random.randint(0, n - width - 1),
            random.randint(0, n - height - 1),
        )

        img[corner[1], corner[0] : corner[0] + width] = 1
        img[corner[1] + height - 1, corner[0] : corner[0] + width] = 1
        img[corner[1] : corner[1] + height, corner[0]] = 1
        img[corner[1] : corner[1] + height, corner[0] + width - 1] = 1
        return img

    @staticmethod
    def generate_cross(img, length_range: List[int]):
        length = random.randint(length_range[0], length_range[1])
        n = img.shape[0]
        center = (
            random.randint(length, n - length - 1),
            random.randint(length, n - length - 1),
        )

        img[center[1], center[0] - length : center[0] + length + 1] = 1
        img[center[1] - length : center[1] + length + 1, center[0]] = 1
        return img

    @staticmethod
    def generate_vertical_bars(img: np.array, length_range: List[int]):
        return DataGenerator.generate_horizontal_bars(img, length_range).T

    @staticmethod
    def generate_horizontal_bars(img: np.array, length_range: List[int]):
        n = img.shape[0]
        nr_bars = 3
        length = random.randint(length_range[0], length_range[1])
        prev_y = -2
        for i in range(nr_bars):
            y = random.randint(prev_y + 2, ((i + 1) * n) // 3 - 1)
            prev_y = y
            x = random.randint(0, n - length)

            img[y, x : x + length] = 1

        return img

    @staticmethod
    def generate_noise(img: np.array, p: float):
        probs = np.random.random(img.shape)
        img[probs < p] = 1
        return img.flatten()


if __name__ == "__main__":
    dg = DataGenerator([5, 10], [5, 10], [5, 10], [5, 10], [3, 7], 3000, 20, 0.01)
    dg.generate_datasets()
