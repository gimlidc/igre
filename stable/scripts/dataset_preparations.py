import click
from stable.dataset.preparation.matrix_3d import crop, merge_image_files


@click.command()
@click.option("-r", "--rectangle", nargs=4, type=int,
              help="Coordinates x0, y0, x1, y1 for cropping. If the rectangle is specified other coordinates are ignored.")
@click.option("-x", "--left-top-x", default=0, help="Crop left top X coordinate.")
@click.option("-y", "--left-top-y", default=0, help="Crop left top Y coordinate.")
@click.option("-w", "--width", default=-1, help="Crop width.")
@click.option("-h", "--height", default=-1, help="Crop height.")
@click.option("--data-key", default="data",
              help="If datastore contains multiple varaibles, here you can specify the key of the pixel intensities.")
@click.argument("data")
def crop(data, left_top_x, left_top_y, width, height, rectangle, data_key):
    """
    Crop of the 3D matrix in form of *.mat file or *.npy.
    Preserves dimensionality and crop along X, Y axes. Axes are expected (width, height, D).
    Produces same type of file (mat/npy).

    DATA - path to input file
    """
    crop(data, left_top_x, left_top_y, width, height, rectangle, data_key, click.echo)


@click.command()
@click.option("--suffix", default="png", help="All images in directory with this suffix will be joined.")
@click.option("--output", default="./ig_dataset.npy", help="Path to generated datastore")
@click.option("--crop/--pad", default=True)
@click.argument("dir_name")
def joindir(dir_name, suffix, output, crop):
    """
    Aggregates images int *.npy file as (width, height, dims) array.
    Images are expected in one folder with specified suffix.
    Files are sorted lexicographically into the array i.e. 3rd dimension of the array corresponds to:
    [file1-dim1, file1-dim2, ... file1-dimD1, file2-dim1, ... fileN-dimDN]

    DIR_NAME - The PATH to data source directory
    """
    merge_image_files(dir_name, suffix, output, crop, click.echo)
