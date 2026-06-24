# Conda-Forge Package Test Devcontainer

Use this devcontainer to validate the released conda-forge package without
mounting the repository's `eo_tools/` source package.

The image intentionally does not install `eo-tools`. This reproduces a clean
user environment where the package is installed manually after opening the
container.

The container mounts the repository's tests, scripts, example data, notebooks,
and `eo_tools_dev/` visualization helpers under `/eo_tools`. `${HOME}/data` is
mounted read-write at `/data`, matching the normal development Compose file.
TiTiler runs alongside the package-test container, and VS Code forwards the
TiTiler and `serve_map(...)` ports.

## Open The Container

From the repository root, open VS Code:

```bash
code .
```

In VS Code, run **Dev Containers: Reopen in Container** and select
**eo_tools conda-forge package test**. To use another host data directory,
edit the `${HOME}/data:/data` mounts in `docker-compose.yml`.

## Install And Verify The Package

The workspace deliberately has no `/eo_tools/eo_tools` source directory.
First confirm that `eo_tools` is not already installed:

```bash
python -c "import eo_tools"
```

Install the released package exactly as a conda-forge user would:

```bash
mamba install -c conda-forge eo-tools=2026.6.0
```

Then confirm that imports resolve to the conda environment:

```bash
python -c "import eo_tools; print(eo_tools.__path__)"
```

Run unit tests and selected real-data scripts directly from the VS Code
terminal:

```bash
pytest -q tests
python scripts/partial_products/test-s1-insar-processor-partial-product.py
```

When a script calls `serve_map(...)`, VS Code forwards port `8000` and opens
the map in the host browser. TiTiler is available on port `8085`.
