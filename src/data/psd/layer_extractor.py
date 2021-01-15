from psd_tools import PSDImage
from os import path


def extract_layers(psd_image_file):
    psd = PSDImage.open(psd_image_file)
    for layer in psd:
        layer.topil.save(psd_image_file[:-len(path.basename(psd_image_file))] + layer.name + '.png', "PNG")


if __name__ == "__main__":
    psd_image_file = '/Users/gimli/Qsync/datasets/Girl with the Pearl Earring/MA-XRF_all_elements.psd'
    extract_layers(psd_image_file)