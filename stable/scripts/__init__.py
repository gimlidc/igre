import click
from dataset.preparation.matrix_3d import crop


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
    Crop of the 3D matrix. Preserves dimensionality and crop along X, Y axes. Produces same type of file.

    DATA - path to input file
    """
    crop(data, left_top_x, left_top_y, width, height, rectangle, data_key, click.echo)
