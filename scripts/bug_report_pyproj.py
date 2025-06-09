# This code works fine in 3.6.1 but randomly fails from 3.7.0
# submitted issue: https://github.com/pyproj4/pyproj/issues/1499
import os
os.environ["PROJ_DEBUG"] = "2"

# use with pyproj 2.5
os.environ["PROJ_NETWORK"] = "ON"

import logging
console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s:%(message)s")
console_handler.setFormatter(formatter)
logger = logging.getLogger("pyproj")
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)

import numpy as np
from pyproj import Transformer
# from pyproj import show_versions
from joblib import Parallel, delayed

# does not work with 2.5
# from pyproj.network import set_network_enabled
# Enable PROJ network temporarily
# set_network_enabled(False)

# show_versions()

# Generate test data
N = 1000
lat = np.random.uniform(-90, 90, N)
lon = np.random.uniform(-180, 180, N)
alt = np.random.uniform(0, 1000, N)

# composite_crs = "EPSG:4326"
composite_crs = "EPSG:4326+3855"
ECEF_crs = "EPSG:4978"       # Earth-Centered Earth-Fixed
chunk_size = 128

# Split into chunks
chunks = [
    (lon[i:i+chunk_size], lat[i:i+chunk_size], alt[i:i+chunk_size])
    for i in range(0, N, chunk_size)
]

# --- Benchmark Functions ---

def run_single_threaded():
    tf = Transformer.from_crs(composite_crs, ECEF_crs, always_xy=True)
    results = tf.transform(lon, lat, alt)
    return results

def run_multi_threaded():
    def transform_chunk(lon_chunk, lat_chunk, alt_chunk):
        tf = Transformer.from_crs(composite_crs, ECEF_crs, always_xy=True)
        return tf.transform(lon_chunk, lat_chunk, alt_chunk)

    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(transform_chunk)(lon, lat, alt) for lon, lat, alt in chunks
    )
    return np.concatenate([r[0] for r in results]), np.concatenate([r[1] for r in results]), np.concatenate([r[2] for r in results])

x1, y1, z1 = run_single_threaded()
x2, y2, z2 = run_multi_threaded()

# --- Check results
def check_close(a, b, label):
    if np.allclose(a, b, atol=1e-6):
        print(f"{label:20s}: Match")
    else:
        print(f"{label:20s}: MISMATCH")

check_close(x1, x2, "Single vs Threads (X)")
check_close(y1, y2, "Single vs Threads (Y)")
check_close(z1, z2, "Single vs Threads (Z)")
# print(z2)