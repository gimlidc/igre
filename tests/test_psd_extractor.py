import tempfile
from psd_tools import PSDImage
import numpy as np

def test_psd2png():
    image = np.random.rand(400, 400, 12) # generate 3 layers RGBA
    tmpfile = tempfile.NamedTemporaryFile(suffix=".psd")
