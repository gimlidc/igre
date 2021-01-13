import stable.modalities.dir_dataset as dataset
import os.path


def test_load_all_images():
    srcdir = os.path.join("tests", "assets")
    data, metadata = dataset.load_all_images(srcdir)
    assert metadata["resolutions"] == [(125, 140)]
    assert data[0].shape[2] == 2
    assert metadata["filenames"][0] == ["mari_magdalena-detail.png", "mari_magdalenaIR-detail.png"]
