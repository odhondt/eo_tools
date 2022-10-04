A Sentinel-1 processor based on SNAP v9.00 and pyroSAR v0.18 for SLC data producing backscatter intensity and/or InSAR coherence and/or Dual pol H/a decomposition

# general information
This processing chain is based on the pyroSAR package. It is capable of distinguishing between ascending and descending orbit and can handle multiple relative orbits.
By default, it creates scenes for each selected polarisation and feature (backscatter intensity, InSAR coherence, Dual pol H/a decomposition).
Mosaicking images from several relative orbits has to be done manually.
The backscatter intensity images are geometrically and radiometrically terrain corrected at gamma nought (Ullmann et al. 2019a, 2019b). The output is either linear or dB.
There is also the possibility to process individual bursts and subswaths.  
Currently, there are still issues with certain projections/ EPSG-codes that SNAP cannot handle properly.










# references
Ullmann, T., Sauerbrey, J., Hoffmeister, D., May, S.M., Baumhauer, R., Bubenzer, O., 2019a. Assessing spatiotemporal variations of sentinel-1 InSAR coherence at different time scales over the atacama desert (Chile) between 2015 and 2018. Remote Sens. 11, 1–22. https://doi.org/10.3390/rs11242960

Ullmann, T., Serfas, K., Büdel, C., Padashi, M., Baumhauer, R., 2019b. Data Processing, Feature Extraction, and Time-Series Analysis of Sentinel-1 Synthetic Aperture Radar (SAR) Imagery: Examples from Damghan and Bajestan Playa (Iran). Zeitschrift für Geomorphol. Suppl. Issues 62, 9–39. https://doi.org/10.1127/zfg_suppl/2019/0524
