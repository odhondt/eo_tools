import pytest
import subprocess
import os
import time
import signal
import numpy as np
from eo_tools.S1.util import remap


def test_remap():
    shape_in = (512, 128)
    img = np.random.rand(*shape_in) + 1j * np.random.rand(*shape_in)

    shape_out = (1024, 2048)
    rr = np.random.rand(*shape_out) * (shape_in[0] - 1)
    cc = np.random.rand(*shape_out) * (shape_in[1] - 1)

    img_out = remap(img, rr, cc, kernel="bicubic")

    assert img_out.shape == rr.shape
    assert img_out.dtype == img.dtype
    assert np.all(~np.isnan(img_out))
