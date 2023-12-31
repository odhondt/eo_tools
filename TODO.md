# Features to implement

## search
- [x] look for scenes intersecting an AOI between 2 dates
- [x] group scenes by orbit and footprint (InSAR processing will be only performed for fully overlapping scenes)
- [x] produce a geo dataframe with scenes that can be displayed on a map to help the user visualizing the different products


## S1 processing
- [x] use a pre-defined graph (one subswath and polarization)
- [x] simple version: takes master and slave assuming they are from the same orbit / slice
- [x] polarization list
- [x] return one file per polarization
- [ ] more complex version: takes a dataframe of filenames and use the groups for coherence / ifgs
- [x] allow processing burst subsets
- [x] change operation order (split before apply orbit)
- [x] add interferogam option
- [x] allow groupByWorkers processing
- [x] geocode independently all IW, stitch in the end with rasterio
- [ ] improved coregistration (ESD, etc)
- [x] make graph more generic by adding placeholders
- [ ] Speckle filtering
    - [ ] for parameter estimation (would need covariance)
    - [ ] for visualization -- channels could be addressed independently

- [x] Split graphs into smaller chunks.  

- [ ] add more parameters, think of a convenient way of doing it (class?)
- [ ] slice assembly (post-process with rio)
- [ ] goldstein filter (optional)
- [ ] gpt options (?)

## S2 processing

- [ ] Optional: apply Sen2Cor on L1C products
- [ ] Test:
    - [ ] L2A processing
    - [x] Baseline < 4
## display

- [x] plot raster on top of folium map, use tile server
- [ ] make HSV composites with phase / coherence / intensity
