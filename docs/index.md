# EO-Tools

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
Examples of the toolbox features can be found here:  

[Example of Sentinel-1 InSAR processing](s1-easy-tops-insar.ipynb)  

[Example of Sentinel-2 search download, mosaic and crop](api.md)  

[Example of DEM download and mosaic and crop](api.md)  


For more advanced usage, read the [API reference](api.md).