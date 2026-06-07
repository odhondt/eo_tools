from types import SimpleNamespace
from unittest.mock import patch

import geopandas as gpd
import pytest
from shapely.geometry import box

from eo_tools.S1.process import (
    _validate_partial_insar_pair,
    _validate_partial_selection,
    process_h_alpha_dual,
    process_insar,
    process_polsar_cov_dual,
    process_slc,
    preprocess_insar_iw,
    preprocess_slc_iw,
)


def _partial_info(subsets=None):
    if subsets is None:
        subsets = {"iw1": ["vv"]}
    return {"subsets": subsets}


def _burst_geometries(bursts, x_offsets=None):
    if x_offsets is None:
        x_offsets = [0] * len(bursts)
    return gpd.GeoDataFrame(
        {
            "burst": bursts,
            "subswath": ["IW1"] * len(bursts),
        },
        geometry=[box(x, 0, x + 1, 1) for x in x_offsets],
        crs="EPSG:4326",
    )


def _partial_swath(min_burst, max_burst, burst_count=9):
    return SimpleNamespace(
        is_partial=True,
        min_burst=min_burst,
        max_burst=max_burst,
        burst_count=burst_count,
        lines_per_burst=10,
        samples_per_burst=20,
        meta={"product": {"swathTiming": {"burstList": {"burst": []}}}},
        compute_burst_overlap=lambda _: 0,
    )


def test_validate_partial_selection_rejects_missing_subswath():
    with pytest.raises(ValueError, match="does not contain IW2"):
        _validate_partial_selection(_partial_info(), ["vv"], ["IW1", "IW2"], "SLC")


def test_validate_partial_selection_rejects_missing_polarization():
    with pytest.raises(ValueError, match="does not contain VH in IW1"):
        _validate_partial_selection(_partial_info(), ["vv", "vh"], ["IW1"], "SLC")


def test_validate_partial_selection_accepts_downloaded_data():
    _validate_partial_selection(
        _partial_info({"iw1": ["vv", "vh"]}), ["vv", "vh"], ["IW1"], "SLC"
    )


def test_validate_partial_insar_pair_rejects_mixed_full_and_partial():
    with pytest.raises(ValueError, match="mixed full and partial pairs"):
        _validate_partial_insar_pair("primary", "secondary", _partial_info(), {})


def test_validate_partial_insar_pair_returns_stored_aoi_for_compatible_pair():
    stored_aoi = box(0, 0, 1, 1)
    with patch(
        "eo_tools.S1.process.read_partial_aoi",
        side_effect=[stored_aoi, box(0, 0, 1, 1)],
    ):
        result = _validate_partial_insar_pair(
            "primary",
            "secondary",
            _partial_info({"iw1": ["vv", "vh"]}),
            _partial_info({"iw1": ["vh", "vv"]}),
        )

    assert result is stored_aoi


def test_validate_partial_insar_pair_rejects_different_stored_aois():
    with patch(
        "eo_tools.S1.process.read_partial_aoi",
        side_effect=[box(0, 0, 1, 1), box(1, 1, 2, 2)],
    ):
        with pytest.raises(ValueError, match="identical stored AOIs"):
            _validate_partial_insar_pair(
                "primary", "secondary", _partial_info(), _partial_info()
            )


def test_validate_partial_insar_pair_rejects_different_subswaths():
    with patch(
        "eo_tools.S1.process.read_partial_aoi",
        side_effect=[box(0, 0, 1, 1), box(0, 0, 1, 1)],
    ):
        with pytest.raises(ValueError, match="same downloaded subswaths"):
            _validate_partial_insar_pair(
                "primary",
                "secondary",
                _partial_info({"iw1": ["vv"]}),
                _partial_info({"iw2": ["vv"]}),
            )


def test_validate_partial_insar_pair_rejects_different_polarizations():
    with patch(
        "eo_tools.S1.process.read_partial_aoi",
        side_effect=[box(0, 0, 1, 1), box(0, 0, 1, 1)],
    ):
        with pytest.raises(ValueError, match="same downloaded polarizations for IW1"):
            _validate_partial_insar_pair(
                "primary",
                "secondary",
                _partial_info({"iw1": ["vv", "vh"]}),
                _partial_info({"iw1": ["vv"]}),
            )


def test_preprocess_insar_rejects_no_common_burst(tmp_path):
    primary = _partial_swath(1, 1)
    secondary = _partial_swath(1, 1)
    primary_geometry = _burst_geometries([1], [0])
    secondary_geometry = _burst_geometries([1], [2])

    with patch(
        "eo_tools.S1.process.S1IWSwath", side_effect=[primary, secondary]
    ), patch(
        "eo_tools.S1.process.get_burst_geometry",
        side_effect=[primary_geometry, secondary_geometry],
    ):
        with pytest.raises(RuntimeError, match="No overlapping bursts"):
            preprocess_insar_iw("primary", "secondary", str(tmp_path))


def test_preprocess_insar_rejects_unavailable_primary_burst(tmp_path):
    primary = _partial_swath(2, 4)
    secondary = _partial_swath(2, 4)
    geometry = _burst_geometries([1], [0])

    with patch(
        "eo_tools.S1.process.S1IWSwath", side_effect=[primary, secondary]
    ), patch(
        "eo_tools.S1.process.get_burst_geometry", side_effect=[geometry, geometry]
    ):
        with pytest.raises(ValueError, match="absent from the partial primary product"):
            preprocess_insar_iw(
                "primary", "secondary", str(tmp_path), min_burst=1, max_burst=2
            )


def test_preprocess_insar_rejects_unavailable_offset_mapped_secondary_burst(tmp_path):
    primary = _partial_swath(2, 4)
    secondary = _partial_swath(2, 4)
    primary_geometry = _burst_geometries([2], [0])
    secondary_geometry = _burst_geometries([3], [0])

    with patch(
        "eo_tools.S1.process.S1IWSwath", side_effect=[primary, secondary]
    ), patch(
        "eo_tools.S1.process.get_burst_geometry",
        side_effect=[primary_geometry, secondary_geometry],
    ):
        with pytest.raises(
            ValueError, match="absent from the partial secondary product"
        ):
            preprocess_insar_iw("primary", "secondary", str(tmp_path))


def test_preprocess_slc_rejects_unavailable_partial_burst(tmp_path):
    slc = _partial_swath(3, 5)

    with patch("eo_tools.S1.process.S1IWSwath", return_value=slc):
        with pytest.raises(ValueError, match="absent from the partial SLC product"):
            preprocess_slc_iw("slc", str(tmp_path), min_burst=2, max_burst=3)


def test_process_insar_uses_effective_partial_aoi_for_geocoding(tmp_path):
    supplied_aoi = box(10, 10, 11, 11)
    stored_aoi = box(0, 0, 1, 1)
    sar_dir = tmp_path / "insar" / "sar"

    with patch(
        "eo_tools.S1.process.prepare_insar", return_value=(str(sar_dir), stored_aoi)
    ) as prepare, patch(
        "eo_tools.S1.process.os.path.isfile", return_value=False
    ), patch(
        "eo_tools.S1.process._child_process"
    ) as child_process:
        result = process_insar(
            "primary",
            "secondary",
            str(tmp_path),
            shp=supplied_aoi,
            pol="vv",
            subswaths=["IW1"],
        )

    assert prepare.call_args.kwargs["shp"] is supplied_aoi
    assert child_process.call_args.args[1]["shp"] is stored_aoi
    assert result == sar_dir.parent


@pytest.mark.parametrize(
    "processor, expected_vars",
    [
        (process_slc, ["amp"]),
        (process_h_alpha_dual, ["H", "alpha", "amp_vv", "amp_vh"]),
        (process_polsar_cov_dual, ["c11", "c22", "c12_real", "c12_imag"]),
    ],
)
def test_slc_based_processor_uses_effective_partial_aoi_for_geocoding(
    tmp_path, processor, expected_vars
):
    supplied_aoi = box(10, 10, 11, 11)
    stored_aoi = box(0, 0, 1, 1)
    sar_dir = tmp_path / processor.__name__ / "sar"

    with patch(
        "eo_tools.S1.process.prepare_slc", return_value=(str(sar_dir), stored_aoi)
    ) as prepare, patch(
        "eo_tools.S1.process.os.path.isfile", return_value=False
    ), patch(
        "eo_tools.S1.process._child_process"
    ) as child_process:
        result = processor(
            "slc",
            str(tmp_path),
            shp=supplied_aoi,
            subswaths=["IW1"],
        )

    assert prepare.call_args.kwargs["shp"] is supplied_aoi
    geocode_args = child_process.call_args.args[1]
    assert geocode_args["shp"] is stored_aoi
    assert geocode_args["var_names"] == expected_vars
    assert result == sar_dir.parent
