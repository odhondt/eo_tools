#! /bin/bash

micromamba init bash
source ~/.bashrc

exec "$@"