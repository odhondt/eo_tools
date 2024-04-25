FROM mambaorg/micromamba

USER root
RUN apt-get update && apt-get install -y \
    software-properties-common
RUN apt-get install -y git python3-pip wget libpq-dev procps gdal-bin openjdk-17-jre-headless

WORKDIR /tmp/
RUN wget https://download.esa.int/step/snap/9.0/installers/esa-snap_sentinel_unix_9_0_0.sh
COPY docker/esa-snap.varfile /tmp/esa-snap.varfile
RUN chmod +x esa-snap_sentinel_unix_9_0_0.sh


RUN /tmp/esa-snap_sentinel_unix_9_0_0.sh -q /tmp/varfile esa-snap.varfile
RUN apt install -y fonts-dejavu fontconfig
COPY docker/update_snap.sh /tmp/update_snap.sh
RUN chmod +x update_snap.sh
RUN /tmp/update_snap.sh

SHELL [ "/bin/bash", "--login", "-c" ]

# needed for opencv
RUN apt-get install -y ffmpeg libsm6 libxext6 libegl1 libopengl0
RUN micromamba shell init --shell=bash --prefix=~/micromamba
RUN source ~/.bashrc
COPY environment.yaml environment.yaml
RUN micromamba create -f  environment.yaml
RUN echo "micromamba activate eo_tools" >> ~/.bashrc

RUN micromamba activate eo_tools