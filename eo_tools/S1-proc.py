def preprocess_insar_iw(
    dir_primary,
    dir_secondary,
    dir_out,
    dir_dem="/tmp",
    iw=1,
    pol="vv",
    min_burst=1,
    max_burst=None,
    # force_write=False,
):
    """Pre-process S1 InSAR subswaths pairs

    Args:
        dir_primary (str): directory containing the primary product of the pair
        dir_secondary (str): _description_
        dir_out (str): output directory
        dir_dem (str, optional): directory where DEMs used for geocoding are stored. Defaults to "/tmp".
        iw (int, optional): subswath index. Defaults to 1.
        pol (str, optional): polarization ('vv','vh'). Defaults to "vv".
        min_burst (int, optional): First burst to process. Defaults to 1.
        max_burst (int, optional): Last burst to process. Defaults to None.
        # force_write (bool, optional): Force overwriting results. Defaults to False.
    """
    # compute LUTS and SLC stack for each burst
    # stitch SLC
    # mosaic LUTs
    pass

