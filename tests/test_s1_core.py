import pytest
from eo_tools.S1.core import S1IWSwath


@pytest.fixture
def create_swath():
    safe_dir = "./data/S1A_IW_SLC__1SDV_20230904T063730_20230904T063757_050174_0609E3_DAA1.SAFE"
    iw = 1
    pol = "vh"
    return S1IWSwath(safe_dir, iw, pol)

def test_s1iwswath_init(create_swath):
    import os

    swath = create_swath

    assert os.path.isfile(swath.pth_tiff)
    assert (
        swath.start_time == "2023-09-04T06:37:31.072288"
    )
    assert swath.lines_per_burst == 1507
    assert swath.samples_per_burst == 23055
    assert swath.burst_count == 9
    assert swath.beta_nought == 2.370000e02
