#!/bin/bash

BASE_DIR=$(pwd)

# Remove previous build artifacts
rm -rf ${BASE_DIR}/build
rm -rf ${BASE_DIR}/dist
rm -rf ${BASE_DIR}/*.egg-info

# Build and install the package
python3 setup.py build bdist_wheel
cd ${BASE_DIR}/dist
pip3 install --force-reinstall *.whl

# Run tests with pytest
# cd ${BASE_DIR}
# pytest -q . || {
#     echo "[ERROR]: Pytest failed";
#     exit 1;
# }
# echo "[INFO]: Pytest success!"

# export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascend910B2C/lib:$LD_LIBRARY_PATH
# msprof op simulator pytest test_cache_unshared_kv.py
