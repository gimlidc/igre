import click
from stable.dataset.preparation.psd_layer_extractor import parse_psd
import scipy.io
import numpy as np
from stable.filepath import change_suffix
from termcolor import colored
from stable.dataset.preparation.matrix_3d import stack_badly_sized_arrays

@click.group(invoke_without_command=False)
def pack():
    pass

@pack.command()
@click.option("-f", "--output-format", help="Output format mat for the *.mat file or *.npy for ndarray")
@click.option("-g", "--grayscale-layers/--fullcoror-layers", default=True,
              help="PSD layers are typically grayscale images, but in PSD stored as RGBA. This flag converts all layers to grayscale.")
@click.argument("psd_file")
def convert(psd_file, output_format: str, grayscale_layers):
    layer_names, arrays = parse_psd(psd_file)
    out = {
        "layer_names": layer_names,
        "images": arrays
    }
    if grayscale_layers:
        arrs = []
        for array, layer_name in zip(arrays, layer_names):
            if len(array.shape) == 2:  # grayscale
                arrs.append(array)
            elif array.shape[2] >= 3:  #
                click.echo(f'{colored(f"Warning:", "yellow")} layer {layer_name} has {colored(array.shape[2], "yellow")} bands and will be compressed to grayscale.')
                arrs.append(
                    (array[:, :, 0] * 0.31 + array[:, :, 1] * 0.58 + array[:, :, 2] * 0.11)\
                    .reshape(array.shape[0], array.shape[1],  1)
                )
    else:
        arrs = arrays

    if output_format.lower() == "mat":
        scipy.io.savemat(change_suffix(psd_file, "mat"), out)
    elif output_format.lower() == "npy":
        hypercube = stack_badly_sized_arrays(layer_names, arrs, True, click.echo)
        for layer_name, image in zip(layer_names, arrays):
            click.echo(f"{layer_name} of shape: {image.shape}")
        np.savez(change_suffix(psd_file, "npy"), hypercube, allow_pickle=True)
    else:
        click.echo("Unsupported output format.")
