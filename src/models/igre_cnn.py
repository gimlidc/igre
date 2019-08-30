from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Flatten, Dense, Reshape, ReLU
from keras.utils import plot_model
import numpy as np
from sklearn.feature_extraction import image


def __build_network(in_dims, out_dims, params):
    model = Sequential()

    model.add(Conv2D(2 * in_dims, kernel_size=5, activation='relu',
                     input_shape=params["patch_size"] + (in_dims,)))
    model.add(MaxPooling2D())
    model.add(Conv2D(in_dims, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(in_dims, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(25))
    inverse_shape = tuple(int(t/np.power(2, 3)) for t in params["patch_size"])
    model.add(Dense(np.prod(inverse_shape) * out_dims))
    model.add(ReLU(1))
    model.add(Reshape(inverse_shape + (out_dims,)))
    model.add(Conv2DTranspose(4 * out_dims, kernel_size=3, strides=(2, 2), padding="same"))
    model.add(Conv2DTranspose(2 * out_dims, kernel_size=3, strides=(2, 2), padding="same"))
    model.add(Conv2DTranspose(out_dims, kernel_size=5, strides=(2, 2), padding="same"))

    if "print" in params and params["print"]:
        plot_model(model, to_file='model.png', show_shapes=True)
    return model


def information_gain(input, target, params=None):
    """
    Method build a CNN according to params. CNN will be composed of several convolutional layers which should convert
    input hyperspectral cube into target hyperspectral cube.
    :param input: hyperspectral cube of input intensities
    :param target: hyperspectral cube of target intensities
    :param params: parameters for building CNN
    :return: tuple
        diff = difference between target and approximation,
        approx = approximated target from input,
        model = builded (and trained) CNN
    """

    if params is None:
        params = {
            "batch": 64,
            "patch_size": (64, 64)
        }

    patches = image.extract_patches_2d(input, params["patch_size"], max_patches=500, random_state=1234)
    targets = image.extract_patches_2d(target, params["patch_size"], max_patches=500, random_state=1234)

    model = __build_network(input.shape[2], target.shape[2], params)
    model.compile(optimizer="adam", metrics=['accuracy'], loss="mean_squared_error")
    model.fit(patches, targets, batch_size=params["batch"], epochs=10)

    approx_patches = model.predict(patches, batch_size=params["batch"])
    approx = image.reconstruct_from_patches_2d(approx_patches, target.shape)
    diff = target - approx
    return diff, approx, model


if __name__ == "__main__":
    # Import all necessary libraries
    import modalities.dir_dataset as dataset
    import cv2
    import matplotlib.pyplot as plt
    import rescale_range as rr

    np.set_printoptions(suppress=True)

    vis = np.asarray(cv2.cvtColor(cv2.imread("/Users/gimli/Qsync/datasets/Suicide_Of_Saul/outputs/visw.png"),
                                  cv2.COLOR_BGR2HSV)).astype(np.float32) / 255.0
    saul, saul_metadata = dataset.load_all_images("/Users/gimli/Qsync/datasets/Suicide_Of_Saul/total_per_tif")

    params = {
        "batch": 64,
        "patch_size": (20, 20)
    }

    diff, approx, model = information_gain(saul[0], vis, params)
    plt.imshow(vis)
    plt.show()

    plt.imshow(rr.rescale_range(approx) * 255)
    plt.show()
