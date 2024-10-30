#!/bin/bash

# 确保脚本以root权限在ubuntu下运行
if [ "$(id -u)" != "0" ]; then
   echo "该脚本必须以root权限运行, run 'sudo bash ./script/install_docker.sh'" 1>&2
   exit 1
fi

echo "start install docker..." 

# Add Docker's official GPG key:
apt-get update
apt-get install ca-certificates curl
apt-get install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update

apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

docker run hello-world