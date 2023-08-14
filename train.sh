#!/bin/bash

# Set environment variables
export WANDB_CACHE_DIR="./wandb/wandb_cache"
export TMPDIR="./wandb/wandb_tmp"
export TMP="./wandb/wandb_tmp"
export TEMP="./wandb/wandb_tmp"

# Call the Python script with all arguments passed to this bash script
python main.py "$@"
