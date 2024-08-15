#!/bin/bash

set -e

export EXPERIMENT_NAME="dqn_cartpole"

python3 run.py -c experiment/${EXPERIMENT_NAME}/run_config.yaml