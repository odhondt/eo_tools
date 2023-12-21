# EO-Tools

A dockerized toolbox for easy programmatic processing of remote sensing imagery from various public sources.

- This project is in its early stages, therefore API is subject to change. 
- Currently, the available features are:
    - Sentinel-1
        - InSAR processor (running SNAP graphs through PyroSAR) computing the coherence, phase and intensities of an interferometric pair of SLC products
        - Writes the result as a geocoded (terrain corrected) COG (Cloud Optimized GeoTIFF) file
        - Displays these rasters in top of a folium map in a jupyter notebook
    - Sentinel-2
        - Tile merging and geocoding
        - Writes any band to COG files
        - Visualization of color composites (Natural RGB, CIR, SWIR, etc) on a folium map
    - All products
        - Search catalog through EODAG
        - Explore products by displaying their footprint on a folium map (custom function)