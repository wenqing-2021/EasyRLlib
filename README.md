# EasyRLlib
EasyRLlib is implemented by Pytorch including the common safe rl algorithms.


## 1. Installation
### 1.1 Install the docker: 
```
bash script/install_docker.sh
```

### 1.2 Install the nvidia-container-toolkit
```
bash script/install_nvidia_container_toolkit.sh
```

### 1.3 enter the env docker (**Make Sure the free space on your disk is more than 20GB**)
```
bash script/enter_dev_env.sh
```

### 1.4 Install the nessecary extension
```
bash script/install_extensions.sh
```
---
Then, you **NEED** to update the environment source, `vim ~/.bashrc`, and add the following command at the end of the file:
```
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
```
Finally, you **NEED** to run: `source ~/.bashrc` in the terminal.

## 2. Run