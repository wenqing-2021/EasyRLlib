#!/bin/bash

# 确保脚本以root权限在ubuntu下运行
if [ "$(id -u)" != "0" ]; then
   echo "该脚本必须以root权限运行, run 'sudo bash ./script/install_nvidia_container_toolkit.sh'" 1>&2
   exit 1
fi

echo "start install nvidia_container_toolkit..." 

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update

apt-get install -y nvidia-container-toolkit

# configure
nvidia-ctk runtime configure --runtime=docker

systemctl restart docker