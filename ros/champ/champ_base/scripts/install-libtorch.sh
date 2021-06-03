#!/bin/bash

ROOT_DIR=$pwd

DIRECTORY="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )/.."

cd $DIRECTORY

wget "https://download.pytorch.org/libtorch/nightly/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip" -O libtorch.zip

unzip "libtorch.zip"

rm "libtorch.zip"

cd $ROOT_DIR