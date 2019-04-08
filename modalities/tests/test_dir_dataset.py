import modalities.dir_dataset as dataset
import os.path


def test_load_all_images():
    srcdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_dir")
    data, metadata = dataset.load_all_images(srcdir)
    assert metadata["resolutions"] == [(125, 140)]
    assert data[0].shape[2] == 2
    assert metadata["filenames"][0] == ["mari_magdalena-detail.png", "mari_magdalenaIR-detail.png"]


if __name__ == "__main__":
    test_load_all_images()
    print("Everything passed")
