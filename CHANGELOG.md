# 2025.1.1

## New feature
- New `dem_name` parameter for SLC and InSAR processors which allows to choose from a list of publicly available DEMs (`nasadem`: SRTM, `alos-dem`: ALOS World 3D, `cop-dem-glo-30` Copernicus DEM 30m and  `cop-dem-glo-90` Copernicus DEM 90m)
- Important note: because of changes in DEM metadata, it is recommended to clean all auto-downloaded DEM files or use `dem_force_download=True` before reprocessing any data after upgrading to this version.

## Bug fixes and improvements
- Add some more tests for S1.core functions (deramping, burst overlap, phi topo,)
- Test script for `cop-dem-glo-30` DEM based insar processing
- Test script comparing SLC processing for all available DEMs
- Fix: use `max_burst_` instead of `max_burst` in burst geometry filtering (`fetch_dem` function)
- Remove unused `dem_profile` output from `geocode_burst`
- DEM files now have a special tag `COMPOSITE_CRS` encoding both lat-lon and vertical reference coordinate system. This tag is added by `fetch_dem`, extracted by `load_dem_coords` and used by `lla_to_ecef` to convert the DEM to ECEF coordinates. 

# 2025.1.0-build2

## Bug Fix
- Fixed a bug in the shadow detector: an offset needed to be added to indices when projecting in the ground geometry to avoid kernel crash. 

## Details
- This is the second build of version `2025.1.0`. The first build (`2025.1.0-1`) involved an update to the Conda recipe without code changes.

# 2025.1.0

## New features
- Radiometric terrain correction (or flattening) is now available in the SLC processor
- Our algorithm differs from the SNAP implementation. This is actually a modified version of the radiometric normalization algorithm described in SNAP's terrain correction documentation (see https://step.esa.int/main/wp-content/help/versions/9.0.0/snap-toolboxes/org.esa.s1tbx.s1tbx.op.sar.processing.ui/operators/RangeDopplerGeocodingOp.html). Two things are different:
    - Instead of the sine of the projected incidence angle, the tangent is computed to comply with the gamma nought convention (as in Small's algorithm).
    - The simulated backscatter is regridded and accumulated in the SAR geometry to account for the many-to-one and one-to-many relationships.
- Shadows are detected and set to NaNs in the normalized image.
- Added tutorial notebooks to demonstrate feature usage.

## Bugfixes and improvements
- Add libgdal jpeg dependency to work with the new version of rasterio.
- Temporary fix: latest pyproj version was introducing a bug in the computation of the DEM XYZ coordinates. Force to use the previous version.
- Refactored both InSAR and SLC processors: we do not use child processes anymore and set the GDAL_CACHEMAX value to a small number. Benchmarks showed smaller memory usage than previously and no performance loss. 
- Fixed some exceptions in the processors (the raise keyword was missing leading the exception to not be triggered).
- Notebooks for conda-forge and docs now use urls to geojson data instead of file paths, making them usable without cloning the github repo.

# 2024.10.1

## New features
- Better S1 coherence estimator
	- Multilooking is performed prior to coherence computation
	- It is applied to each individual term in the coherence expression
	- This way larger sample sizes can be collected even with small boxcar windows
	- It avoids averaging the coherence itself (theoretically sounder)
	- Binary erosion is applied to mitigate discontinuities at subswath borders
	- Default multilook is still `[1, 4]` but coherence window size is now `[3, 3]`
## Bugfixes
- Fixed an error with DEM download due to Planetary Computer now requiring signed urls

## Other
- Upgraded dependencies

# 2024.10.0

## New features
- Sentinel-1 zipped products are now handled (updated all notebooks accordingly).
- InSAR processing of partially overlapping products, for instance between S1A and S1B, as long as share the same orbit.

## Other
- Refactored block processing utility to handle larger overlap.
- Goldstein filter now uses different block size (32) and overlap results in smaller final block size.
- Added Etna notebook to illustrate partial overlap in InSAR (docker version only)
- Simple script that shows all S1 products located in a directory on a map (not part of the conda package)

# 2024.9.2

## New features
- New function `S1.process.goldstein`: the Goldstein interferometric filter is an adaptive method that allows to reduce noise on phase.
- Tutorial notebooks on how to use the function
- All file names like `"ifg*"` are now treated as interferograms by `S1.process.geocode_and_merge_iw`. This way users can compare several phase filtering methods easily. 

## Bugfixes
- Correct options for orbit retrieval `['POE', 'RES']` using `pyrosar`
- In `S2.process_tiles`, `baseline > 4` replaced by `baseline >= 4`
- In `S1.process.coherence` output is tiled like input

## Misc
- `S1.process.coherence` with manually fixed chunk size is faster.

# 2024.9.1

## New feature -- Sentinel-1 custom pipelines
- New `S1.process.apply_multilook` function to apply multilook on a tiff in the SAR geometry.
- Multilooking may now also be applied in `S1.process.coherence`, `S1.process.amplitude` and `S1.process.interferogram` rather than in `S1.process.geocode_and_merge_iw`. 
- These functions change the transform in the GeoTIFF files. `S1.process.sar2geo` is now aware of this and applies automatic rescaling. This allows the user to call `S1.process.sar2geo` after multilooking an image and define custom processing chains.
- In this spirit, two new helper functions allow to apply any user defined function to all subswaths and polarizations present in the output product directory.
- Therefore, custom processing chains of type `S1.process.prepare_insar` -> _custom processing_  -> `S1.process.geocode_and_merge_iw` can be easily defined by the user. 
- The same can be applied to `S1.process.prepare_slc` for single images.
- Notebooks showing an example of such custom pipeline implementing both coherent and incoherent change detection have been added to the docs and the example folders.

## Bugfixes
- Corrected the `S1.util.presum` function to avoid overwriting the input array, reformatted docstrings

## Other
- Added test script for `S1.process.preprocess_insar_iw`
- Automatically downloaded DEM (for S1 processing) file names are now based on unique hashes instead of being based on product/subswath/polarization. 
- This has several advantages:
    - It makes filenames shorter
    - A new file is created every time a DEM related parameter changes (e.g. `dem_upsampling`) instead of overwriting the old one
    - It enables re-using DEM files across functions
    - When processing both polarizations, the DEM is not downloaded twice under a different name anymore
    - The parameter `force_dem_download` is now set to `False` by default


# 2024.9.0

## New features
- New SLC processor to compute and geocode amplitude from a single SLC product.
- Calibration can now be either set to `sigma` or `beta` nought in both InSAR and SLC processors. For the InSAR processor it is useful when amplitude is computed. 
- Both `beta` and `sigma` nought calibration factors are computed by the new `S1IWSwath.calibration_factor` function. Beta normalization is not applied anymore by `S1IWSwath.read_burst`.
- Processing functions `process_*` now have a `cal_type` parameter to control the calibration type.
- The docker version can now be built with or without SNAP (connect to the folder with SSH and select `rebuild without cache`).

## Bug fixes
- `dem_force_download` warning: it is now displayed when the value is set to `False`.
- the `warp_kernel` parameter is now propagated in all processing functions.

## Other
- Unit tests for `S1IWSwath.calibration_factor` and `S1IWSwath.read_burst`.
- Test script and example notebook for `process_slc`.
- More consistent logging messages.

# 2024.8.1

- Refactored the Sentinel-1 `_process_bursts` function of the InSAR processor
- Geocoding lookup tables (LUT) now uses a common DEM 
- No more LUT merging after processing, safer handling of burst overlap to avoid gaps
- The processor uses a new function `S1IWSwath.fetch_dem` that downloads the DEM for all processed bursts 
- The DEM is now upsampled after download, not at terrain correction stage 
- Burst geometries are computed at `S1IWSwath` initialization  and stored as a class member 
- Burst geometries are now used instead of GCPs
- DEM cropping uses a buffer around the burst geometry
- Fixed a bug in the `load_metadata` function: `rfi` files are filtered out from the metadata file search 
- Handling of non-float NaNs in DEM (at coordinate reading) 
- `range_doppler` function skips NaN values
- Optimization: Using `cache=True` and `nogil=True` in `coregister_fast` and `range_doppler`
- Optimization: avoid the creation of multiple small arrays by `coregister_fast` results in speedup
- Bug fix: in the  s1-core-demo notebooks, burst index testing has been corrected in the interferogram warping cell
- Better test data
- Wrote a test for the insar processor using this data

# 2024.8.0 

This version contains bugfixes. `rioxarray` was not producing valid output COGs because it required extra options at raster creation.  

Rendering of geocoded products on a map should be much faster.  

Change list:
- Fix invalid COGs: use COG driver after merging
- Fix the geocoding of real-valued data by also using COG in `sar2geo`
- Using better COG creation options
    - ZSTD compression instead of DEFLATE
    - overview and projections with nearest neighbor resampling
    - using num_threads="all_cpus" to make compression faster


# 2024.7.0

## Code improvements and some bugfixes:
- Burst Ids are checked only when available in the product metadata
- Improved code clarity by renaming some variables, explicitly naming positional parameters
- Added user friendly exceptions to check product validity
- Cleaned up some old comments
- Added numexpr to environments for upcoming optimizations
- Test data for full processor
- Optional directory for orbit files (default is `\tmp`)
- Added and improved existing logging messages
- Refactored `resampling` to use `remap`
- Fixed bug in `resampling`: using output profile of upsampled DEM instead of original DEM 
- Changed example notebooks accordingly

# 2024.6.3

# added some parameters for more flexibility
- add iw and pol subsets options to geocoding
- default behaviour of the processor will change:
- insar outputs will only be computed for subsets
- inside the processor, geocoding still looks for
any subset
- outside the processor the new options take effect

# bugfix
- fix error when `burstId` is not present in the xml and
display a warning

# 2024.6.2

- New InSAR processor for full products
	- Processing, geocoding and merging subswaths in one function call
	- Computation of coherence and / or interferogram
	- Optional computation of amplitudes
	- Selection of subswath and polarization subsets
	- Optional crop of an area of interest (to save computation, only intersecting bursts are processed)
	- Access to intermediate files in the SAR geometry. These can be further processed prior to geocoding (for instance, speckle filtering or incoherent change detection of amplitudes may be applied) 
	- Standalone functions to geocode and merge any raster in the SAR geometry
	- Cloud Optimized GeoTIFF (COG) geocoded output file which can be displayed in GIS software or uploaded to tile servers
- Improved internals
	- Some memory intensive functions are now using child processes to ensure memory from large objects will be released after processing. This ensures an overall lower memory footprint.

# 2024.6.1

## build 0 on conda-feedstock:
- removed code that was not supposed to be used in the conda version
- example notebooks can be found in `notebooks-cf`

## build 1 on conda-feedstock:
- fix missing dependencies causing eodag to crash (uvicorn has missing requirements)

# 2024.6.0

- initial release