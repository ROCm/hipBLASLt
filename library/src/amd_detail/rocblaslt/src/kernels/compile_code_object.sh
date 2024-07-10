#!/bin/bash
sources=$1
archs=$2
dest=$3
rocm_path="${ROCM_PATH:-/opt/rocm}"
hipcc_path="${rocm_path}/bin/hipcc"
$hipcc_path "$sources" --offload-arch="${archs}" --genco -O0 -g -o "$dest"