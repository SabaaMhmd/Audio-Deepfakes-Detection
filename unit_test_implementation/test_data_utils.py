import sys
sys.path.append("software_implementation/aasist")

from data_utils import pad, pad_random
from software_implementation.aasist.data_utils import (
    pad,
    pad_random
)

def test_pad_output_length():
    x = np.random.randn(1000)
    output = pad(x, max_len=96000)

    assert len(output) == 96000


def test_pad_random_output_length():
    x = np.random.randn(5000)
    output = pad_random(x, max_len=96000)

    assert len(output) == 96000


def test_pad_keeps_exact_length():
    x = np.random.randn(96000)
    output = pad(x, max_len=96000)

    assert len(output) == 96000
