import json
from unittest.mock import patch

import geopandas as gpd
import pytest
import yaml
from shapely.geometry import Point, box, mapping

from eo_tools.S1.core import (
    S1IWSwath,
    read_partial_aoi,
    read_partial_download_info,
)


def _write_geojson(path, geometries):
    features = [
        {"type": "Feature", "properties": {}, "geometry": mapping(geometry)}
        for geometry in geometries
    ]
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}),
        encoding="utf-8",
    )


def _write_partial_manifest(path, manifest):
    (path / "partial_download.yml").write_text(
        yaml.safe_dump(manifest), encoding="utf-8"
    )


def _partial_swath(min_burst=3, max_burst=5):
    swath = S1IWSwath.__new__(S1IWSwath)
    swath.min_burst = min_burst
    swath.max_burst = max_burst
    return swath


def test_read_partial_download_info_returns_empty_for_regular_product(tmp_path):
    assert read_partial_download_info(tmp_path) == {}


def test_read_partial_download_info_reads_manifest(tmp_path):
    manifest = {"aoi_file": "partial_aoi.geojson", "subsets": {"iw1": {"vv": {}}}}
    _write_partial_manifest(tmp_path, manifest)

    assert read_partial_download_info(tmp_path) == manifest


def test_read_partial_aoi_requires_manifest_aoi_file(tmp_path):
    with pytest.raises(ValueError, match="does not specify required AOI metadata"):
        read_partial_aoi(tmp_path, {"subsets": {}})


def test_read_partial_aoi_requires_existing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="AOI metadata file is missing"):
        read_partial_aoi(tmp_path, {"aoi_file": "partial_aoi.geojson"})


@pytest.mark.parametrize(
    "geometries, error",
    [
        ([], "exactly one feature"),
        ([box(0, 0, 1, 1), box(2, 2, 3, 3)], "exactly one feature"),
        ([Point(0, 0)], "one non-empty Polygon"),
    ],
)
def test_read_partial_aoi_rejects_invalid_geometry(tmp_path, geometries, error):
    aoi_file = tmp_path / "partial_aoi.geojson"
    _write_geojson(aoi_file, geometries)

    with pytest.raises(ValueError, match=error):
        read_partial_aoi(tmp_path, {"aoi_file": aoi_file.name})


def test_read_partial_aoi_returns_polygon(tmp_path):
    expected = box(0, 0, 1, 1)
    aoi_file = tmp_path / "partial_aoi.geojson"
    _write_geojson(aoi_file, [expected])

    result = read_partial_aoi(tmp_path, {"aoi_file": aoi_file.name})

    assert result.equals(expected)


def test_partial_swath_init_rejects_missing_subswath_polarization(tmp_path):
    safe_path = tmp_path / "S1A_IW_SLC__TEST.partial.SAFE"
    safe_path.mkdir()
    aoi_file = safe_path / "partial_aoi.geojson"
    _write_geojson(aoi_file, [box(0, 0, 1, 1)])
    _write_partial_manifest(
        safe_path,
        {
            "aoi_file": aoi_file.name,
            "subsets": {"iw1": {"vv": {"file": "measurement/test.tiff"}}},
        },
    )

    with pytest.raises(ValueError, match="does not contain subswath IW2"):
        S1IWSwath(safe_path, iw=2, pol="vv")


def test_validate_available_burst_uses_partial_range():
    swath = _partial_swath()

    swath._validate_available_burst(3)
    swath._validate_available_burst(5)

    with pytest.raises(ValueError, match="must be between 3 and 5"):
        swath._validate_available_burst(2)
    with pytest.raises(ValueError, match="must be between 3 and 5"):
        swath._validate_available_burst(6)


@pytest.mark.parametrize(
    "method_name, args",
    [
        ("read_burst", (2,)),
        ("geocode_burst", ("dem.tif", 2)),
        ("deramp_burst", (2,)),
        ("calibration_factor", (2,)),
    ],
)
def test_partial_burst_methods_reject_unavailable_burst(method_name, args):
    swath = _partial_swath()

    with pytest.raises(ValueError, match="must be between 3 and 5"):
        getattr(swath, method_name)(*args)


def test_fetch_dem_defaults_to_partial_max_burst(tmp_path):
    swath = _partial_swath()
    swath.gdf_burst_geom = gpd.GeoDataFrame(
        {"burst": [3, 4, 5]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
        crs="EPSG:4326",
    )

    with patch.object(
        swath, "_validate_available_burst", wraps=swath._validate_available_burst
    ) as validate, patch("eo_tools.S1.core.os.path.exists", return_value=True):
        dem_file = swath.fetch_dem(min_burst=3, dem_dir=str(tmp_path))

    assert dem_file.startswith(str(tmp_path))
    assert [call.args[0] for call in validate.call_args_list] == [3, 5]
