FROM mambaorg/micromamba:latest

USER root

RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    wget \
    libpq-dev \
    procps \
    gdal-bin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/conda_init

COPY environment.yaml .

# Create the environment
RUN micromamba create -y -n eo_tools -f environment.yaml && \
    micromamba clean --all --yes

# Make the environment the default Python
ENV PATH=/opt/conda/envs/eo_tools/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV MAMBA_DEFAULT_ENV=eo_tools
RUN echo "alias conda=micromamba" >> /root/.bashrc
RUN echo "micromamba activate eo_tools" >> /root/.bashrc

WORKDIR /