#!/bin/bash

NAME=mri_tabular_docker
MAIN_DIR=$(realpath "`dirname $0`/../")
SSH_PORT=2773
SSD_DIR="/home/lin/dataset/mri-ssd"

cd "`dirname $0`"
NETWORK="host"

docker run -it --rm --gpus all -e PLATFORM=$ENV_PLATFORM \
    -p ${SSH_PORT}:22 \
    --network $NETWORK \
    -v $MAIN_DIR:/home/work \
    -v $SSD_DIR:/home/ssd \
    --shm-size 16G \
    --name $NAME \
    $NAME \
    bash
