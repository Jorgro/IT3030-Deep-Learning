from typing import List
import numpy as np
import random

import pickle
from flags import SHOW_IMAGES
import matplotlib.pyplot as plt

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

        self.dataset_file_name = "dataset.pickle"

    @staticmethod
    def get_2d_img(img):
        n = int(img.shape[0] ** (1 / 2))
        return img.reshape((n, n))

    def generate_datasets(self):

        # Generate the datasets
        datasets = {"size: ": self.image_size}

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
        with open("dataset2.pickle", "wb") as handle:
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

    @staticmethod
    def show_images(images):
        _, axarr = plt.subplots(2, 5)
        for i in range(2):
            for j in range(5):
                axarr[i, j].imshow(DataGenerator.get_2d_img(images[i * 5 + j]))
        plt.show()


if __name__ == "__main__":
    dg = DataGenerator(
        rec_width_range=[6, 15],
        rec_height_range=[6, 15],
        ver_length_range=[6, 15],
        hor_length_range=[6, 15],
        cro_length_range=[3, 8],
        nr_images=1200,
        image_size=20,
        noise=0.01,
    )
    dg.generate_datasets()

    if SHOW_IMAGES:
        with open("dataset2.pickle", "rb") as file:
            data = pickle.load(file)
            x_train = np.array(data["x_train"])
        images = np.random.choice(x_train.shape[0], 10, replace=False)
        DataGenerator.show_images(x_train[images])
