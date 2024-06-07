# EO-Tools documentation

EO-Tools is a python toolbox that is currently able to search, download and process Sentinel-1 InSAR pairs, download and mosaic Sentinel-2 tiles and download various publicly available DEM (Digital Elevation Models). The S1 processor can compute phase, amplitude and coherence in the SAR geometry and reproject them in a geographic coordinate system. 

## Install

- The package comes in two flavors
    - A conda package that contains the main functionality (Sentinel-1 InSAR, Sentinel-2 tile mosaic and DEM download)
    - A docker version (for more advanced users) that additonally works with a TiTiler server for interactive visualization in the notebooks
    - The legacy SNAP based processor is only available in the docker version.

### Conda install

- It is recommended to first create a conda environment to avoid package conflicts
- You need to have `conda` installed (or `mamba` / `micromamba`)
- Then the package can be installed with these commands (replace `conda` by `mamba` or `micromamba` if needed):

```bash
conda env create -n eo_tools
conda activate eo_tools
conda install conda-forge::eo_tools
```

## Usage
Examples can be found here:  

- [Sentinel-1 InSAR processing.](s1-easy-tops-insar.ipynb)  

- [Sentinel-2 search download, mosaic and crop.](discover-and-process-s2.ipynb)  

- [DEM download and mosaic and crop.](download-dem.ipynb)  

Additionally, more example notebooks can be found in the [github](https://github.com/odhondt/eo_tools) repository of the package.

The different functions are documented in the [API reference](api.md).