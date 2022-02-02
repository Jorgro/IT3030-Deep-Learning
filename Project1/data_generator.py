from enum import Enum
import os
import shutil
from skimage.draw import random_shapes
from skimage.io import imsave
import numpy as np


class ShapeClass(Enum):
    CIRCLE = 0
    RECTANGLE = 1
    TRIANGLE = 2
    CROSS = 3


class DataGenerator:
    def __init__(
        self,
        nr_images: int,
        image_size: int,
        noise: float,
        train_size: float = 0.70,
        test_size: float = 0.20,
        val_size: float = 0.10,
    ):
        self.image_size = image_size
        self.nr_images = nr_images
        self.noise = noise

        self.paths = ["train", "test", "val"]
        self.sizes = [
            round(train_size * nr_images),
            round(test_size * nr_images),
            round(val_size * nr_images),
        ]

        self.dataset_path = "dataset"

    def generate_datasets(self):
        if os.path.exists(self.dataset_path):
            shutil.rmtree(self.dataset_path)
        os.mkdir(self.dataset_path)
        for path, size in zip(self.paths, self.sizes):
            os.mkdir(f"{self.dataset_path}/{path}")
            self.generate_dataset(f"{self.dataset_path}/{path}", size)

    def generate_dataset(self, path: str, size: int):
        result, _ = random_shapes(
            (20, 20),
            num_channels=1,
            max_shapes=1,
            shape="rectangle",
            random_seed=0,
            min_size=10,
            max_size=20,
            intensity_range=(0, 0),
        )
        prob = 0.01
        probs = np.random.random(result.shape)
        result[probs < prob] = 0
        # result[result > 0] = 255
        imsave(f"{path}/rectangle1.png", result)

    @staticmethod
    def generate_rectangle(img):
        pass

    @staticmethod
    def generate_circle(img):
        pass

    @staticmethod
    def generate_vertical_bars(img):
        pass

    @staticmethod
    def generate_horizontal_bars(img):

        return img


if __name__ == "__main__":
    dg = DataGenerator(1000, 0.01)
    dg.generate_datasets()

