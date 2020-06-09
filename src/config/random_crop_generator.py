import numpy as np

class RandomCropGenerator:

    def __init__(self, image_size, count=10, sample_size=(600, 600), safe_boundary=(30, 30, 30, 30)):
        np.random.seed(123456789)
        self.x = np.random.randint(safe_boundary[0], image_size[0] - sample_size[0] - safe_boundary[2],
                                   int(np.ceil(np.sqrt(count))))
        self.y = np.random.randint(safe_boundary[1], image_size[1] - sample_size[1] - safe_boundary[3],
                                   int(np.ceil(np.sqrt(count))))
        self.sample_size = sample_size
        self.count = count

    def get_crop(self, index):
        if index >= self.count:
            raise IndexError("Index out of bounds of generated array")
        (x, y) = np.unravel_index(index, (int(np.ceil(np.sqrt(self.count))), int(np.ceil(np.sqrt(self.count)))))
        return int(self.x[x]), int(self.y[y]), int(self.sample_size[0]), int(self.sample_size[1])
