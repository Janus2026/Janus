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
from utils.config import Config

from templates.common_matmul_template import CommonMatmulTemplate
from templates.launch_map_template import LaunchMapTemplate

if __name__ == "__main__":

    kernel_info = {}

    os.makedirs(Config.WRAPPER_CODE_PATH, exist_ok=True)
    CommonMatmulTemplate.gen_code("CommonMatmulKernel", "common_matmul_kernel", 0, "half", kernel_info)
    LaunchMapTemplate.gen_code(kernel_info)