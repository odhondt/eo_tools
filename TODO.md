# Features to implement

## search
- [ ] look for scenes intersecting an AOI between 2 dates
- [ ] group scenes by orbit and footprint (coherence will be only computed for fully overlapping scenes)


## processing
- [ ] use a pre-defined graph (one subswath and polarization)
- [ ] simple version: takes master and slave assuming they are from the same orbit / slice
- [ ] polarization list
- [ ] return one file per polarization
- [ ] more complex version: takes a dataframe of filenames and use the groups for coherence / ifgs
- [x] allow processing burst subsets
- [x] change operation order (split before apply orbit)
- [ ] add interferogam option
- [x] allow groupByWorkers processing
- [x] geocode independently all IW, stitch in the end with rasterio
- [ ] improved coregistration (ESD, etc)

## display

- [ ] plot raster on top of folium map, use tile server
