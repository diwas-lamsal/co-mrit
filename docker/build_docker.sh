#!/bin/bash
NAME=mri_tabular_docker
cd "`dirname $0`"

docker build . -t $NAME
