FROM mambaorg/micromamba

USER root
RUN apt-get update && apt-get install -y \
    software-properties-common
RUN apt-get install -y git python3-pip wget libpq-dev procps gdal-bin
# needed for opencv
# RUN apt-get install -y ffmpeg libsm6 libxext6 libegl1 libopengl0



SHELL [ "/bin/bash", "--login", "-c" ]

RUN micromamba shell init --shell=bash --root-prefix=~/micromamba
RUN source ~/.bashrc
COPY environment.yaml environment.yaml
RUN micromamba create -f  environment.yaml
RUN echo "micromamba activate eo_tools" >> ~/.bashrc
RUN echo "alias conda='micromamba'" >> ~/.bashrc

RUN micromamba activate eo_tools