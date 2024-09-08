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