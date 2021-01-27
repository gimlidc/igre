import click
import os
import numpy as np
import imageio
from datetime import datetime
from termcolor import colored
from stable.information_gain.pixelwise import information_gain
import scipy.io
from dataset.preprocessing.rescale_range import rescale_range

@click.group()
def ig():
    pass

@ig.command()
@click.option(
    "-i",
    "--in-dims",
    required=True,
    help="Comma separated integers corresponding with INPUT dimensions for ANN in data_file. Indexed from 0."
)
@click.option(
    "-o",
    "--out-dims",
    required=True,
    help="Comma separated integers corresponding with OUTPUT dimensions for ANN in data_file. Indexed from 0."
)
@click.option(
    "-l",
    "--layers",
    required=False,
    help="Comma separated widths of ANN"
)
@click.option("--output-dir", default=f"./ig-out-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
@click.option("--training-set-size", default=50000, help="Number of pixels used for training.")
@click.option("--data-key", default="data",
              help="If datastore contains multiple varaibles, here you can specify the key of the pixel intensities.")
@click.argument('data_file')
def gain(data_file, in_dims, out_dims, layers, output_dir, training_set_size, data_key):
    """
    Computation of information gain per modality defined in OUT_DIMS with respect to information content of IN_DIMS.
    Tool produces two images per each output dimension:
    - {OUT_DIM}-approx.png which contains extrapolated information from IN_DIMS to output modality
    - {OUT_DIM}-gain.png which contains extra information present in the output modality

    DATA_FILE - Registered dataset in format of 3D matrix (width x height x modality). If you have input and output
    data in separate files use `ig joindir` of `ig joinfiles` before.
    """
    click.echo(colored(f"Computation of information gain for {data_file}.", "green"))
    click.echo(f"\tDimensions conversion {in_dims} => {out_dims}")
    click.echo(f"\tOutput folder: {output_dir}")

    inds = [int(dim) for dim in in_dims.split(",")]
    ouds = [int(dim) for dim in out_dims.split(",")]

    # Load data file and split it into input and output for IG
    suffix = os.path.splitext(data_file)[1][1:]
    data = None
    if suffix == "npy": # numpy array
        data = np.load(data_file)
    elif suffix == "mat":
        matfile = scipy.io.loadmat(data_file)
        datakeys = [key for key in matfile.keys() if not key.startswith("__")]
        if len(datakeys) == 1:
            data = rescale_range(matfile[datakeys[0]])
        else:
            if data_key in datakeys:
                data = rescale_range(matfile[data_key])
            else:
                click.echo(colored(f"Multiple variables stored in specified matfile: {datakeys}. "
                                   f"Please use --data-key option to choose one.", "red"))
    if data is None:
        click.echo(colored(f"Unsupported file format: {suffix}.", "red"))
        click.echo("Supported formats - *.mat, *.npy")
        return 1
    visible = np.array(data[:, :, inds], dtype=float)
    target = np.array(data[:, :, ouds], dtype=float)

    # Run IG
    gain, approx, model = information_gain(visible, target, layers, training_set_size)

    # Create output directory (if necessary)
    click.echo(f"Creating directory {output_dir}")
    os.makedirs(output_dir)

    # store all files
    outfiles = []
    for idx, dim in enumerate(ouds):
        # Src
        imageio.imwrite(os.path.join(output_dir, f"{dim}-original.png"), (target[:, :, idx] * 255).astype(np.uint8))
        # Gain output
        outfiles.append(os.path.join(output_dir, f"{dim}-gain.png"))
        imageio.imwrite(outfiles[-1], gain[:, :, idx])
        # Approx output
        outfiles.append(os.path.join(output_dir, f"{dim}-approx.png"))
        imageio.imwrite(outfiles[-1], (approx[:, :, idx] * 255).astype(np.uint8))

    # Finish gracefully
    click.echo(colored(f"Processed files:", "green"))
    [click.echo("\t" + file) for file in outfiles]
    click.echo(colored("successfully written.", "green"))
