import click
import os
import numpy as np
import imageio
from datetime import datetime
from termcolor import colored
from src.models.ig import information_gain
import scipy.io
from src.data.rescale_range import rescale_range


@click.group(chain=True)
def ig():
    pass


@click.group(chain=True)
def datajoin():
    pass


@datajoin.command()
@click.option(
    "--output",
    default="ig_dataset.npy",
    help="Path to output file where all images will be joined"
)
@click.argument(
    "files",
    # help="Any number of files to join"
)
def datafiles(files):
    click.echo(f"These files will be joined into ND array: {files}")


@datajoin.command()
@click.option("--suffix", default="png", help="All images in directory with this suffix will be joined.")
@click.option("--output", default="./ig_dataset.npy", help="Path to generated datastore")
@click.option("--crop/--pad", default=True)
@click.argument("dir_name")
def joindir(dir_name, suffix, output, crop):
    """
    Generates one file from all input images. Images are expected in one folder with specified suffix.

    DIR_NAME - The PATH to data source directory
    """
    if not os.path.isdir(dir_name):
        click.echo("ERROR: directory not found")

    imgs = []
    data_files = sorted(os.listdir(dir_name))
    click.echo("Loading files ...")
    for file in data_files:
        if file[-len(suffix):] == suffix:
            img = imageio.imread(os.path.join(dir_name, file))
            if len(img.shape) == 2:
                img = img.reshape(img.shape + (1,))
            imgs.append(img)
            click.echo(f"\t{file} OK, shape: {imgs[-1].shape}")

    shapes = np.array([np.array(img.shape) for img in imgs])
    size_matches = True
    for dim_name, dim in zip(["width", "height"], [0, 1]):
        if np.min(shapes[:,dim]) != np.max(shapes[:,dim]):
            click.echo(colored(
                f"Warning: Image {dim_name}s mismatch: {np.min(shapes[:, dim])} "
                f"in {data_files[np.argmin(shapes[:, dim])]} "
                f"vs. {np.max(shapes[:, dim])} in {data_files[np.argmax(shapes[:, dim])]}"), color="yellow")
            if crop:
                click.echo(colored(f"Data will be cropped to min {dim_name} size."), color="green")
            else:
                click.echo(colored(f"Data will be padded with zeros to match max size.", color="green"))
            size_matches = False

    if crop or size_matches:
        width = np.min(shapes[:, 0])
        height = np.min(shapes[:, 1])
        out = np.zeros((width, height, np.sum(shapes[:, 2])))
        start_dim = 0
        for img in imgs:
            out[:, :, start_dim:start_dim + img.shape[2]] = img[:width, :height, :]
            start_dim += img.shape[2]

        np.save(output, out)
    else:  # pad option
        out = np.zeros((np.max(shapes[:, 0]), np.max(shapes[:, 1]), np.sum(shapes[:, 2])))
        start_dim = 0
        for img in imgs:
            out[:img.shape[0], :img.shape[1], start_dim:start_dim + img.shape[2]] = img
            start_dim += img.shape[2]
        np.save(output, out)

    click.echo(colored(f"Data successfully saved into {output} with shape {out.shape}.", color="green"))


@click.group()
def process():
    pass


@process.command()
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


ig = click.CommandCollection(sources=[datajoin, process])

if __name__ == '__main__':
    ig()
