import pytest
import numpy as np
from pathlib import Path

from eo_tools.auxils import block_process, get_burst_geometry

# Example processing function
def simple_process_fn(block, multiplier=1):
    """An example processing function that multiplies the block."""
    return block * multiplier

@pytest.mark.parametrize(
    "input_array, block_size, overlap, multiplier, expected_output, expected_exception",
    [
        # Basic functionality with overlap (the output should be the input multiplied by the multiplier)
        (
            np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12],
                      [13, 14, 15, 16]]),
            (2, 2),
            (1, 1),
            2,
            np.array([[ 2,  4,  6,  8],
                      [10, 12, 14, 16],
                      [18, 20, 22, 24],
                      [26, 28, 30, 32]]),
            None
        ),
        # Non-overlapping windows
        (
            np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12],
                      [13, 14, 15, 16]]),
            (2, 2),
            (0, 0),
            3,
            np.array([[ 3,  6,  9, 12],
                      [15, 18, 21, 24],
                      [27, 30, 33, 36],
                      [39, 42, 45, 48]]),
            None
        ),
        # Dimension mismatch
        (
            np.random.rand(4, 4),
            (2, 2, 2),
            (1, 1),
            2,
            None,
            ValueError
        ),
        # Processing with a multiplier
        (
            np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12],
                      [13, 14, 15, 16]]),
            (2, 2),
            (1, 1),
            4,
            np.array([[ 4,  8, 12, 16],
                      [20, 24, 28, 32],
                      [36, 40, 44, 48],
                      [52, 56, 60, 64]]),
            None
        ),
    ]
)
def test_block_process(input_array, block_size, overlap, multiplier, expected_output, expected_exception):
    if expected_exception:
        with pytest.raises(expected_exception):
            block_process(input_array, block_size, overlap, simple_process_fn, multiplier=multiplier)
    else:
        output_array = block_process(input_array, block_size, overlap, simple_process_fn, multiplier=multiplier)
        print("Output array:")
        print(output_array)
        print("Expected array:")
        print(expected_output)
        assert np.array_equal(output_array, expected_output)


def test_get_burst_geometry_accepts_path_objects():
    safe_path = Path(
        "data/S1/S1A_IW_SLC__1SDV_20230904T063730_20230904T063757_050174_0609E3_DAA1.SAFE"
    )

    burst_geom = get_burst_geometry(safe_path, "IW1", "VH")

    assert not burst_geom.empty
    assert set(burst_geom["subswath"]) == {"IW1"}
