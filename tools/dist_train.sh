#!/usr/bin/env bash

CONFIG=$2
GPUS=4
PORT=$1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}



#python -m torch.distributed.launch --nproc_per_node=1 --master_port=29701 ./tool/train.py './configs/scrfd/scrfd_500m_zennas.py'
