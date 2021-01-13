import click
import scipy.io
import numpy as np
import os
from termcolor import colored

@click.command()
@click.option("-r", "--rectangle", nargs=4, type=int, help="Coordinates x0, y0, x1, y1 for cropping. If the rectangle is specified other coordinates are ignored.")
@click.option("-x", "--left-top-x", default=0, help="Crop left top X coordinate.")
@click.option("-y", "--left-top-y", default=0, help="Crop left top Y coordinate.")
@click.option("-w", "--width", default=-1, help="Crop width.")
@click.option("-h", "--height", default=-1, help="Crop height.")
@click.option("--data-key", default="data",
              help="If datastore contains multiple varaibles, here you can specify the key of the pixel intensities.")
@click.argument("data")
def crop(data, left_top_x, left_top_y, width, height, rectangle, data_key):
    """
    Crop of the 3D matrix. Preserves dimensionality and crop along X, Y axes. Produces same type of file.

    DATA - path to input file
    """

    suffix = os.path.splitext(data)[1][1:]
    matrix = None
    if suffix == "npy":  # numpy array
        matrix = np.load(data)
    elif suffix == "mat":
        matfile = scipy.io.loadmat(data)
        datakeys = [key for key in matfile.keys() if not key.startswith("__")]
        if len(datakeys) == 1:
            data_key = datakeys[0]
        if data_key in datakeys:
            matrix = matfile[data_key]
        else:
            click.echo(colored(f"Multiple variables stored in specified matfile: {datakeys}. "
                               f"Please use --data-key option to choose one.", "red"))
    if matrix is None:
        click.echo(colored(f"Unsupported file format: {suffix}.", "red"))
        click.echo("Supported formats - *.mat, *.npy")
        return 1

    if rectangle is None:
        x0 = left_top_x
        x1 = left_top_x + width
        y0 = left_top_y
        y1 = left_top_y + height
    else:
        x0 = rectangle[0]
        y0 = rectangle[1]
        x1 = rectangle[2]
        y1 = rectangle[3]
    click.echo(f"Rectangle coordinates [{x0},{y0},{x1},{y1}]")

    base = os.path.basename(data)
    output_file = f"{data[:-len(base)]}{base[:-len(suffix)-1]}-crop.{suffix}"
    if suffix == "mat":
        out = {}
        click.echo(matrix[x0:x1, y0:y1, :].shape)
        out[data_key] = matrix[x0:x1, y0:y1, :]
        scipy.io.savemat(output_file, out)
    elif suffix == "npy":
        np.save(output_file, matrix[x0:x1, y0:y1, :])
    click.echo(colored(f"Output file: {output_file} successfully written.", "green"))
