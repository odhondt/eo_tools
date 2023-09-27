# Features to implement

## search
- [ ] look for scenes intersecting an AOI between 2 dates
- [ ] group scenes by orbit and footprint (coherence will be only computed for fully overlapping scenes)


## processing
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
- [x] improved coregistration (ESD, etc)
- [x] make graph more generic by adding placeholders

- [x] Split graphs into smaller chunks.  

- add some parameters
- slice assembly (post-process with rio)
- goldstein filter (optional)
- gpt options (?)
- write class (?)

## display

- [ ] plot raster on top of folium map, use tile server
