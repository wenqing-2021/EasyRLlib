#!/bin/bash

set -e

# 定义实验列表
PARAMS=("dqn_cartpole"
        "ddpg_Pendulum-v1"
        "ppo_BipedalWalker-v3"
        "ppo_cartpole"
        "sac_BipedalWalker-v3"
        "sacd_cartpole"
        )

for param in "${PARAMS[@]}"; do
    echo "Executing $PROGRAM with parameter: $param"
    python3 run.py -c experiment/${param}/run_config.yaml
done