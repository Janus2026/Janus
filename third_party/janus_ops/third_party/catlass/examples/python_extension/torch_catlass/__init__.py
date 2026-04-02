#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import os
import sysconfig
import torch
import torch_npu

__all__ = []

def _load_depend_libs():
    PYTHON_PKG_PATH=sysconfig.get_paths()['purelib']
    TORCH_LIB_PATH=os.path.join(PYTHON_PKG_PATH,"torch/lib")
    TORCH_NPU_LIB_PATH=os.path.join(PYTHON_PKG_PATH,"torch_npu/lib")
    os.environ['LD_LIBRARY_PATH'] = f"{os.environ['LD_LIBRARY_PATH']}:{TORCH_LIB_PATH}:{TORCH_NPU_LIB_PATH}"
    
_load_depend_libs()

from torch_catlass._C import *