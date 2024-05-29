import pytest
import subprocess
import os
import time
import signal
import numpy as np
from eo_tools.util import TileServerManager
from eo_tools.S1.util import remap


def test_remap():
    shape_in = (512, 128)
    img = np.random.rand(*shape_in) + 1j * np.random.rand(*shape_in)

    shape_out = (1024, 2048)
    rr = np.random.rand(*shape_out) * (shape_in[0] - 1)
    cc = np.random.rand(*shape_out) * (shape_in[1] - 1)

    img_out = remap(img, rr, cc, kernel="bicubic")

    assert img_out.shape == rr.shape
    assert img_out.dtype == img.dtype
    assert np.all(~np.isnan(img_out))


# Define a high-numbered port to reduce the chance of conflicts
TEST_PORT = 49152


def test_start_stop():
    print("test_start_stop")
    # Stop the server if it's already running
    TileServerManager.stop()
    # Start the server
    TileServerManager.start(port=TEST_PORT, timeout=30)
    assert TileServerManager._get_server_pid() is not None
    # Stop the server
    TileServerManager.stop()
    assert TileServerManager._get_server_pid() is None


def test_start_existing_port():
    print("test_existing")
    # Stop the server if it's already running
    TileServerManager.stop()
    # Start the server on an existing port
    TileServerManager.start(port=TEST_PORT, timeout=30)
    assert TileServerManager._get_server_pid() is not None
    # Try to start another server on the same port
    with pytest.raises(RuntimeError) as excinfo:
        TileServerManager.start(port=TEST_PORT)
    assert "running" in str(excinfo.value) and str(TEST_PORT) in str(excinfo.value)
    # Stop the server
    TileServerManager.stop()


def test_start_timeout():
    print("test_timeout")
    # Stop the server if it's already running
    TileServerManager.stop()
    # Start the server with a very short timeout
    with pytest.raises(Exception) as excinfo:
        TileServerManager.start(port=TEST_PORT, timeout=0.5)
    assert "Timeout reached" in str(excinfo.value)


def test_stop_not_running():
    print("test_stop_not_running")
    # Ensure the server is not running
    TileServerManager.stop()
    # Attempt to stop the server when it's not running
    TileServerManager.stop()
    assert TileServerManager._get_server_pid() is None


def test_cleanup():
    print("test_cleanup")
    # Ensure the server is not running
    TileServerManager.stop()
    # Check that the cleanup function properly stops the server
    TileServerManager.start(port=TEST_PORT, timeout=30)
    assert TileServerManager._get_server_pid() is not None
    # Ensure the server is stopped for subsequent tests
    TileServerManager.stop()
    assert TileServerManager._get_server_pid() is None
