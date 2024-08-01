#!/bin/bash

echo "enter dev env..."
current_dir=$(pwd)
base_name=$(basename "$current_dir")

echo "current dir is : $current_dir"
echo "base name is :   $base_name"

docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $current_dir:/home/$base_name nvcr.io/nvidia/pytorch:23.08-py3