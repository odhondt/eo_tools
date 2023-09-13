# Features to implement

## search
- [ ] look for scenes intersecting an AOI between 2 dates
- [ ] group scenes by orbit and footprint (coherence will be only computed for fully overlapping scenes)

## start from a predefined graph

## processing
- [ ] check if parameters for coregistration are correct (dem, interpolation...)
- [ ] check if full footprint correspond (not only orbits)
- [ ] allow processing burst subsets
- [ ] change operation order (split before apply orbit)
- [ ] add interferogam option
- [ ] allow groupByWorkers processing
- [ ] geocode independently all IW, stitch in the end with rasterio

## display

- [ ] plot raster on top of folium map
