#!/bin/bash

# Set this to avoid fragmentation
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

# Argument 1: Sweep ID, no default value so it's mandatory
SWEEP_ID=${1}

# Argument 2: Device index, default is 0
DEVICE_INDEX=${2:-0}

# Argument 3: Maximum number of jobs to execute, default is 10000
MAX_JOBS=${3:-10000}

# Running the specified command with the given and default arguments
dysweep_run_resume --package train \
                   --function dysweep_compatible_run \
                   --run_additional_args gpu_index:$DEVICE_INDEX \
                   --config meta_configurations/training/base.yaml \
                   --sweep_id $SWEEP_ID \
                   --count $MAX_JOBS \
                   --mark_preempting True
