#!/bin/bash

### ---------------------------------------------------------------------------
###                              BOOK KEEPING VARS
### ---------------------------------------------------------------------------
pytorch_ver=`pip show torch | grep Version | awk '{print $2}'`
rocm_ver=`dpkg -l rocm-core | grep rocm-core | awk '{print $3}'`
miopen_ver=`dpkg -l miopen-hip | grep miopen-hip | awk '{print $3}'`
DAY=$(date +%d-%m-%y)
HEADER="Date:$DAY\nPytorch,rocm,miopen\n$pytorch_ver,$rocm_ver,$miopen_ver\n"

### ---------------------------------------------------------------------------
###                           USER CONFIGURABLE VARS
### ---------------------------------------------------------------------------
export PROFILE_DIR="results/"
export PYTORCH_MIOPEN_SUGGEST_NHWC=1 # Turn on NHWC (Not Default for Pytorch)
#export MIOPEN_ENABLE_LOGGING_CMD=1 # View Pytorch MIOPEN commands 

### ---------------------------------------------------------------------------
###                           PROFILE OPTIONS
### ---------------------------------------------------------------------------
#PROFILE='rpd'
#ROFILE='torch'
PROFILE='none'

### ---------------------------------------------------------------------------
###                           SIZE OPTIONS
### ---------------------------------------------------------------------------
BATCH_SIZE="1 2 4 64 128 256"
INPUT_SIZE="14 28 56 224"
STRIDE="1 2"
INPUT_CHANNELS="192"
OUTPUT_CHANNELS="192"
KERNEL_SIZE="1 3"
PADDING_SIZE="1 2"

## Full Scan 
#BATCH_SIZE="1 2 4 11 16 21 32 48 64 128 256 512"
#INPUT_SIZE="14 28 56 112 224 513 3000"
#STRIDE="1 2 4"
#INPUT_CHANNELS="3 64 192 256 384 512 640 960 1024 1280 2048 2560"
#OUTPUT_CHANNELS="60 64 192 256 384 512 640 1024 1280 2048 8192"
#KERNEL_SIZE="1 2 3 5 7"
#PADDING_SIZE="0 1 2"

### ---------------------------------------------------------------------------
###                                  EXECUTION
### ---------------------------------------------------------------------------

afo conv --batch_size_list "$BATCH_SIZE" --input_size_list "$INPUT_SIZE" --kernel_size_list "$KERNEL_SIZE" --stride_list "$STRIDE" --input_channels_list "$INPUT_CHANNELS" --output_channels_list "$OUTPUT_CHANNELS" --padding_list "$PADDING_SIZE" --verbose --profile $PROFILE
