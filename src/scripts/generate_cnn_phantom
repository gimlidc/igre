import click
import json
import pickle

from src.data.phantom.materials import extract_pigments, create_cnn_phantom, get_pigment_dist


@click.command()
@click.option('-c', '--config', required=True, help='Config, for further details', type=click.File('rb'))
@click.option('-o', '--output_name', required=True, help='Name of outputfile', type=str)
def generate_cnn_phantom(config, output_name):
    args = json.load(config)
    assert 'pad' in args.keys()
    assert 'extractor_args' in args.keys()
    assert 'phantom_args' in args.keys()
    pad = args['pad']
    # Separate pigments
    pigments = extract_pigments(**args['extractor_args'])
    # Calc distribution
    pigments_dist = []
    for pigment in pigments:
        no_draw, draw = pigment['no_draw'], pigment['draw']
        pigments_dist.append({"draw": get_pigment_dist(draw, pad),
                              "no_draw": get_pigment_dist(no_draw, pad)})
    # Create phantom
    cnn_phantoms = create_cnn_phantom(pigments_dist,
                                      sample_shape=(3, 3, 32),
                                      **args['phantom_args'])
    # Include config
    cnn_phantoms['config'] = args

    with open(f'{output_name}.pickle', 'wb') as f:
        pickle.dump(cnn_phantoms, f)

if __name__ == "__main__":
    generate_cnn_phantom()
