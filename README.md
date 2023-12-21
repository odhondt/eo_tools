# EO-Tools

A dockerized toolbox for easy programmatic processing of remote sensing imagery from various public sources.

- This project is in its early stages, therefore API is likely to change. 
- Currently, the available features are:
    - Sentinel-1
        - InSAR processor (running SNAP graphs through PyroSAR) computing the coherence, phase and intensities of an interferometric pair of SLC products
        - Write the result as a geocoded (terrain corrected) COG (Cloud Optimized GeoTIFF) file
        - Display these rasters in top of a folium map in a jupyter notebook
    - Sentinel-2
        - Tile merging and geocoding
        - Write any band to COG files
        - Visualization of color composites (Natural RGB, CIR, SWIR, etc) on a folium map
    - All products
        - Search catalog (using EODAG) and download products
        - Explore products by displaying their footprint on a folium map (custom function)
- Example notebooks can be found in the `notebooks/` folder