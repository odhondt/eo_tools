# EO-Tools [![Conda Version](https://img.shields.io/conda/vn/conda-forge/eo-tools.svg)](https://anaconda.org/conda-forge/eo-tools)  
EO-Tools is a pure python toolbox that is currently able to search, download and process Sentinel-1 InSAR pairs, download and mosaic Sentinel-2 tiles and download various publicly available DEM (Digital Elevation Models). The S1 processor can compute phase, amplitude and coherence in the SAR geometry and reproject them in a geographic coordinate system. Example notebooks demonstrating the different features are located in the notebooks-cf folder of the github repository.

## New since version 2024.6.3
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

Here are examples of amplitude, phase and coherence computed using this framework:
<p float="left">
    <img src="https://raw.githubusercontent.com/odhondt/eo_tools/main/data/ex_amp.png" width="220">
    <img src="https://raw.githubusercontent.com/odhondt/eo_tools/main/data/ex_phi.png" width="220">
    <img src="https://raw.githubusercontent.com/odhondt/eo_tools/main/data/ex_coh.png" width="220">
</p>

## Overview
- Currently, the available features are:
    - Sentinel-1
        - New standalone InSAR processor (see previous section)
        - Legacy InSAR processor (running SNAP graphs through PyroSAR) computing the coherence, phase and intensities of an interferometric pair of SLC products
        - Write the result as a geocoded (terrain corrected) COG (Cloud Optimized GeoTIFF) file
        - Display these rasters on top of a folium map in a jupyter notebook
    - Sentinel-2
        - Tile merging and geocoding
        - Write any band to COG files
        - Visualization of color composites (Natural RGB, CIR, SWIR, etc) on a folium map
    - DEM
        - Automatically downloads and crops a DEM given a geometry 
    - All products
        - Search catalog (using EODAG) and download products
        - Explore products by displaying their footprint on a folium map (custom function)
        - Show remote and local images on top of folium maps in the notebook
- Example notebooks can be found in the `notebooks/` folder

## Install & quick start

- The package comes in two flavors
    - A conda package that contains the main functionality (Sentinel-1 InSAR, Sentinel-2 tile mosaic and DEM download)
    - A docker version (for more advanced users) that additonally works with a TiTiler server for interactive visualization in the notebooks
    - The legacy SNAP based processor is only available in the docker version.

### Conda install (recommended)

- It is recommended to first create a conda environment to avoid package conflicts
- You need to have `conda` installed (or `mamba` / `micromamba`)
- Then the package can be installed with these commands (replace `conda` by `mamba` or `micromamba` if needed):

```bash
conda env create -n eo_tools
conda activate eo_tools
conda install conda-forge::eo-tools 
```

### Docker install

- It works as a dev container for VSCode. 
    - Clone the github repository into the location of your choice.
    - Volumes paths can (and should) be changed in `docker-compose.yml`.
    - After opening the main directory, VSCode should detect the devcontainer file and ask to build the container. Once the container is running, the example notebooks in the `notebooks` directory can be used.
- Alternatively, it should also be possible to start the container from the main directory with `docker-compose up -d` in a terminal and attach to the container with any editor supporting docker.

### Getting started

- Please make sure `jupyter` is installed in your environment
- Example jupyter notebooks demonstrate the different features
- For conda use the notebooks in the `notebooks-cf` directory of the github repository
- For docker use the notebooks in the `notebooks` directory of the github repository

## Notice

- This project was originally forked from: https://github.com/eo2cube/s1_processor/, however since 99% of the code is now original, I have detached the fork.
- Visualization functions are using TiTiler https://developmentseed.org/titiler/
- Product discovery and download are using EODAG https://github.com/CS-SI/eodag
- The old S1 processor uses pyroSAR https://github.com/johntruckenbrodt/pyroSAR which executes graphs with ESA's SNAP software https://github.com/senbox-org
