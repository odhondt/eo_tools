FROM mambaorg/micromamba:debian12-slim
# FROM mambaorg/micromamba


USER root
RUN apt-get update && apt-get install -y \
    software-properties-common
RUN apt-get install -y git python3-pip wget libpq-dev procps gdal-bin

WORKDIR /tmp/conda_init
COPY environment.yaml .
# Create environment
RUN micromamba create -y -n eo_tools -f environment.yaml
# Make the environment the default python
ENV PATH=/opt/conda/envs/eo_tools/bin:$PATH
# Optional but recommended
ENV PYTHONUNBUFFERED=1
WORKDIR /