import pytest
from pathlib import Path
import os
import numpy as np
from eo_tools.S1.core import S1IWSwath


def test_s1iwswath_init():
    import os
    # print(os.get_cwd())
    safe_dir = "./data/S1A_IW_SLC__1SDV_20230904T063730_20230904T063757_050174_0609E3_DAA1.SAFE"
    iw = 1
    pol = "vh"

    # Initialize the S1IWSwath object
    swath = S1IWSwath(safe_dir, iw, pol)

    # Assertions
    assert os.path.isfile(swath.pth_tiff)
    assert (
        swath.start_time == "2023-09-04T06:37:31.072288"
    )  # Replace with actual start time from your metadata
    assert swath.lines_per_burst == 1507
    assert swath.samples_per_burst == 23055
    assert swath.burst_count == 9
    assert swath.beta_nought == 2.370000e02
