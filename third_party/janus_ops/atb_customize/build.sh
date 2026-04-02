#!/bin/bash
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

export CODE_ROOT=$(pwd)
export CACHE_DIR=$CODE_ROOT/build
OUTPUT_DIR=$CODE_ROOT/output
THIRD_PARTY_DIR=$CODE_ROOT/3rdparty

USE_CXX11_ABI=""
COMPILE_OPTIONS=""
CMAKE_BUILD_TYPE=Release
BUILD_OPTION_LIST="help default unittest clean"
BUILD_CONFIGURE_LIST=("--use_cxx11_abi=0" "--use_cxx11_abi=1" "--debug")

function fn_build()
{
    if [ "${ASCEND_HOME_PATH}" == "" ]; then
        echo "error: build failed because ASCEND_HOME_PATH is null, please source cann set_env.sh first."
        exit 1
    fi
    if [ "${ATB_HOME_PATH}" == "" ]; then
        echo "error: build failed because ATB_HOME_PATH is null, please source nnal set_env.sh first."
        exit 1
    fi
    if ! grep -q "8.5.0" "${ATB_HOME_PATH}/../../version.info"; then
        echo "error: build failed because this function only support Ascend-cann-atb : 8.5.0."
        exit 1
    fi
    fn_load_3rdparty_for_compile
    [[ ! -d "$CACHE_DIR" ]] && mkdir $CACHE_DIR
    cd $CACHE_DIR
    echo "COMPILE_OPTIONS:$COMPILE_OPTIONS"
    cmake $CODE_ROOT $COMPILE_OPTIONS
    cmake --build . --parallel $COMPILE_VERBOSE
    cmake --install .
}

function fn_build_googletest()
{
    if [ -d "$THIRD_PARTY_DIR/googletest/lib" ]; then
        return 0
    fi
    [[ ! -d "$THIRD_PARTY_DIR" ]] && mkdir $THIRD_PARTY_DIR
    cd $THIRD_PARTY_DIR
    [[ ! -d "googletest" ]] && git clone --branch v1.13.0 --depth 1 https://github.com/google/googletest.git
    cd googletest
    sed -i '34d' CMakeLists.txt
    rm -rf build && mkdir build && cd build
    if [ "$USE_CXX11_ABI" == "ON" ]
    then
        sed -i '33 a add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)' ../CMakeLists.txt
    else
        sed -i '33 a add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)' ../CMakeLists.txt
    fi
    cmake .. -DCMAKE_INSTALL_PREFIX=$THIRD_PARTY_DIR/googletest -DCMAKE_SKIP_RPATH=TRUE -DCMAKE_CXX_FLAGS="-fPIC"
    cmake --build . --parallel $(nproc)
    cmake --install . > /dev/null
    [[ -d "$THIRD_PARTY_DIR/googletest/lib64" ]] && cp -rf $THIRD_PARTY_DIR/googletest/lib64 $THIRD_PARTY_DIR/googletest/lib
    echo "Googletest is successfully installed to $THIRD_PARTY_DIR/googletest"
}

function fn_load_nlohmann_json()
{
    if [ -d "$THIRD_PARTY_DIR/nlohmannJson" ]; then
        return 0
    fi
    cd $THIRD_PARTY_DIR
    # git clone --branch v3.11.3 --depth 1 https://github.com/nlohmann/json.git
    git clone --branch v3.11.3 --depth 1 https://gitcode.com/GitHub_Trending/js/json.git
    mv json nlohmannJson
}

function fn_load_mki()
{
    if [ -d "$THIRD_PARTY_DIR/Mind-KernelInfra" ]; then
        return 0
    fi
    cd $THIRD_PARTY_DIR
    git clone -b br_release_cann_8.5.0_20260527 --depth 1 https://gitcode.com/cann/ascend-boost-comm.git Mind-KernelInfra
}

function fn_load_atb()
{
    if [ -d "$THIRD_PARTY_DIR/ascend-transformer-boost" ]; then
        return 0
    fi
}

function fn_load_compiler()
{
    if [ -d "$THIRD_PARTY_DIR/compiler" ]; then
        return 0
    fi
    mkdir -p $THIRD_PARTY_DIR/compiler
    CCEC_COMPILER_DIR=$THIRD_PARTY_DIR/compiler/ccec_compiler
    TIKCPP_DIR=$THIRD_PARTY_DIR/compiler/tikcpp
    ln -s $ASCEND_HOME_PATH/tools/ccec_compiler $CCEC_COMPILER_DIR
    ln -s $ASCEND_HOME_PATH/compiler/tikcpp $TIKCPP_DIR
}

function fn_load_3rdparty_for_compile()
{
    [[ ! -d "$THIRD_PARTY_DIR" ]] && mkdir $THIRD_PARTY_DIR
    fn_load_nlohmann_json
    fn_load_mki
    fn_load_atb
    fn_load_compiler
}

function fn_run_unittest()
{
    if [ "$USE_CXX11_ABI" == "ON" ]; then
        export LD_LIBRARY_PATH=$OUTPUT_DIR/ops_customize/cxx_abi_1/lib:$LD_LIBRARY_PATH
    else
        export LD_LIBRARY_PATH=$OUTPUT_DIR/ops_customize/cxx_abi_0/lib:$LD_LIBRARY_PATH
    fi
    $OUTPUT_DIR/bin/customize_blockcopy_test
    $OUTPUT_DIR/bin/custom_paged_attention_test
}

function fn_init_env()
{
    res=$(python3 -c "import torch" &> /dev/null || echo "torch_not_exist")
    if [ "$res" == "torch_not_exist" ]; then
        echo "Warning: Torch is not installed!"
        [[ "$USE_CXX11_ABI" == "" ]] && USE_CXX11_ABI=ON
    fi
    if [ "$USE_CXX11_ABI" == "" ]; then
        if [ $(python3 -c 'import torch; print(torch.compiled_with_cxx11_abi())') == "True" ]; then
            USE_CXX11_ABI=ON
        else
            USE_CXX11_ABI=OFF
        fi
    fi
    export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
    export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
    export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
    export PYTORCH_NPU_INSTALL_PATH="$(python3 -c 'import torch, torch_npu, os; print(os.path.dirname(os.path.abspath(torch_npu.__file__)))')"
    export LD_LIBRARY_PATH=$PYTORCH_INSTALL_PATH/../torch.libs:$LD_LIBRARY_PATH

    echo "PYTHON_INCLUDE_PATH=$PYTHON_INCLUDE_PATH"
    echo "PYTHON_LIB_PATH=$PYTHON_LIB_PATH"
    echo "PYTORCH_INSTALL_PATH=$PYTORCH_INSTALL_PATH"
    echo "PYTORCH_NPU_INSTALL_PATH=$PYTORCH_NPU_INSTALL_PATH"
    echo "USE_CXX11_ABI=$USE_CXX11_ABI"
}


function fn_main()
{
    if [[ "$BUILD_OPTION_LIST" =~ "$1" ]]; then
        if [[ -z "$1" ]]; then
            arg1="default"
        else
            arg1=$1
            shift
        fi
    else
        cfg_flag=0
        for item in ${BUILD_CONFIGURE_LIST[*]}; do
            if [[ "$1" =~ $item ]]; then
                cfg_flag=1
                break 1
            fi
        done
        if [[ "$cfg_flag" == 1 ]]; then
            arg1="default"
        else
            echo "argument $1 is unknown, please type build.sh help for more imformation"
            exit 1
        fi
    fi

    until [[ -z "$1" ]]
    do {
        arg2=$1
        case "${arg2}" in
        "--use_cxx11_abi=1")
            USE_CXX11_ABI=ON
            ;;
        "--use_cxx11_abi=0")
            USE_CXX11_ABI=OFF
            ;;
        "--debug")
            CMAKE_BUILD_TYPE=Debug
            ;;
        esac
        shift
    }
    done

    [[ ! -d "$OUTPUT_DIR" ]] && mkdir $OUTPUT_DIR
    fn_init_env

    COMPILE_OPTIONS="${COMPILE_OPTIONS} -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE -DUSE_CXX11_ABI=$USE_CXX11_ABI"
    case "${arg1}" in
        "default")
            fn_build
            ;;
        "clean")
            [[ -d "$CACHE_DIR" ]] && rm -rf $CACHE_DIR
            [[ -d "$OUTPUT_DIR" ]] && rm -rf $OUTPUT_DIR
            [[ -d "$THIRD_PARTY_DIR" ]] && rm -rf $THIRD_PARTY_DIR
            echo "clean all build history"
            ;;
        "unittest")
            echo "start ops_customize unittest"
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DBUILD_CUSTOMIZE_OPS_TEST=ON"
            fn_build_googletest
            fn_build
            fn_run_unittest
            ;;
        *)
        echo "Usage: "
        echo "run build.sh help|default|unittest|clean| --use_cxx11_abi=0|--use_cxx11_abi=1|--debug"
        ;;
    esac
}

fn_main "$@"
