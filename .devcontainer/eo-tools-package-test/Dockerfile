FROM condaforge/miniforge3:latest

RUN mamba create -y -n eo_tools_package_test -c conda-forge \
        pytest \
        ipykernel \
        jupyter \
    && mamba clean --all --yes

ENV PATH=/opt/conda/envs/eo_tools_package_test/bin:$PATH
ENV CONDA_DEFAULT_ENV=eo_tools_package_test
ENV PYTHONUNBUFFERED=1

WORKDIR /eo_tools
