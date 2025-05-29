<p float="left">
    <img src="https://raw.githubusercontent.com/odhondt/eo_tools/main/data/eo-tools.svg" width="300">
</p>

# A python Earth Observation toolbox

[![Conda Version](https://img.shields.io/conda/vn/conda-forge/eo-tools.svg)](https://anaconda.org/conda-forge/eo-tools) [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/eo_tools.svg)](https://anaconda.org/conda-forge/eo_tools)  
EO-Tools is a pure python toolbox that is currently able to search, download and process Sentinel-1 InSAR pairs, geocode and apply terrain correction to Sentinel-1 SLC products, download and mosaic Sentinel-2 tiles and download various publicly available DEM (Digital Elevation Models). The S1 processor can compute phase, amplitude and coherence in the SAR geometry and reproject them in a geographic coordinate system. Example notebooks demonstrating the different features are located in the notebooks-cf folder of the github repository.

**Important:** read about version 2025.2.0 breaking change in version notes (`CHANGELOG.md`).

Here are examples of EO-Tools outputs showing amplitude, coherence and inteferometric phase of a 2023 earthquake in Morocco,
<p float="left">
    <img src="https://raw.githubusercontent.com/odhondt/eo_tools/main/data/ex_amp.png" width="220">
    <img src="https://raw.githubusercontent.com/odhondt/eo_tools/main/data/ex_coh.png" width="220">
    <img src="https://raw.githubusercontent.com/odhondt/eo_tools/main/data/ex_phi.png" width="220">
</p>
a comparison between Sentinel-1 amplitude, coherence, change map using two dates in 2021 and Sentinel-2 RGB image over the city of Berlin in Germany,
<p float="left">
    <img src="https://raw.githubusercontent.com/odhondt/eo_tools/main/data/amp_vh_berlin.png" width="350">
    <img src="https://raw.githubusercontent.com/odhondt/eo_tools/main/data/coh_vh_berlin.png" width="350">
    <img src="https://raw.githubusercontent.com/odhondt/eo_tools/main/data/change_berlin.png" width="350">
    <img src="https://raw.githubusercontent.com/odhondt/eo_tools/main/data/s2_berlin.png" width="350">
</p>
and the comparison between Beta nought calibration and terrain flattening using Copernicus DEM on Sentinel-1 data over Etna.
<p float="left">
    <img src="https://raw.githubusercontent.com/odhondt/eo_tools/main/data/ex_etna_beta0.png" width="300">
    <img src="https://raw.githubusercontent.com/odhondt/eo_tools/main/data/ex_etna_flattened.png" width="300">
</p>

## Overview
- Currently, the available features are:
    - Sentinel-1
        - Interferometric processing of Sentinel-1 pairs, including TOPS processing steps like azimuth deramping, DEM assisted coregistration, Range-Doppler terrain correction and Enhanced Spectral Diversity. Individual bursts can be processed as well as full products and cropped to any area of interest provided by the user
        - Amplitude geocoding of SLC Sentinel-1 images, with Beta or Sigma Nought or terrain flattening calibration.
        - Ability to apply processing in the SAR geometry and further project the results in a geographic coordinate systems using lookup-tables.
        - Writing the result as a geocoded (terrain corrected) COG (Cloud Optimized GeoTIFF) file
        - Displaying these rasters on top of a folium map in a jupyter notebook (docker version only)
    - Sentinel-2
        - Tile merging and geocoding
        - Write any band to COG files
        - Visualization of color composites (Natural RGB, CIR, SWIR, etc) on a folium map (docker version only)
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

### Conda install (recommended)

- It is recommended to first create a conda environment to avoid package conflicts
- You need to have `conda` installed (or `mamba` / `micromamba`)
- Then the package can be installed with these commands (replace `conda` by `mamba` or `micromamba` if needed):

```bash
conda create -n eo_tools
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