# EasyRLlib

EasyRLlib is implemented by Pytorch including the common safe rl algorithms.

## 1. Installation

### 1.1 Install the docker:

```
bash script/tools/install_docker.sh
```

### 1.2 Install the nvidia-container-toolkit

```
bash script/tools/install_nvidia_container_toolkit.sh
```

### 1.3 Enter the env docker

```
bash script/tools/enter_dev_env.sh
```

### 1.4 Install the nessecary extension

```
bash script/tools/install_extensions.sh
```

---

Then, you **NEED** to update the environment source, `vim ~/.bashrc`, and add the following command at the end of the file:

```
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
```

Finally, you **NEED** to run: `source ~/.bashrc` in the terminal.

## 2. Run

### 2.1 Specific your config

you can update the configure as the following format yaml and put it in the experiment path.

```yaml
# this file is the configuration file for the training process
env_config:
  env_name: "CartPole-v1" # set the env_name

train_config:
  seed: 1
  epochs: 25
  total_steps: 1000000 # normally is 1e6
  num_envs: 4
  batch_size: 32 # if use on-policy algorithm, the batch_size is the same as the buffer_size
  buffer_size: 256 # if use the on-policy algorithm, the buffer_size would be int(total_steps/epochs/num_envs)

  off_policy_train_config:
    update_every: 25 # update the network every steps
    random_explor_steps: 20000 # random explore the env before the training
    soft_update_every: 100 # update the target network every steps

  on_policy_train_config:
    update_times: 80 # set the update_times, update the network times
    max_ep_len: 1000 # set the max_ep_len, the max length of the episode

agent_config_path: "src/config/agent_config/ppo.yaml"
exp_name: "DQN"
save_path: "output/"
device: "cpu" # set the device, "cuda" or "cpu"
```

you may define the agent config where is put in the root: `src/config/agent_config`

### 2.2 Run the training

you can change the run_config as you changed:

```bash
python3 run.py -c src/config/run_config.yaml
```

## 3. Surpport Agent List

| Agent Name | Discrete Action | Continous Action |
| :--------: | :-------------: | :--------------: |
|    DQN     |     **YES**     |      **NO**      |
|    SAC     |     **YES**     |     **YES**      |
|    PPO     |     **YES**     |     **YES**      |
|    DDPG    |     **NO**      |     **TODO**     |
|    TD3     |     **NO**      |     **TODO**     |
|    A2C     |    **TODO**     |     **TODO**     |
