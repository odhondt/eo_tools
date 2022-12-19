FROM mambaorg/micromamba

USER root
RUN apt-get update && apt-get install -y \
    software-properties-common
RUN apt-get install -y git python3-pip wget procps

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

COPY environment.yaml environment.yaml
RUN micromamba install -y --file environment.yaml && \
     micromamba clean --all --yes


WORKDIR /app/
COPY . /app/
RUN source ~/.bashrc \
 && python -m pip install .


COPY docker/entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]