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

class Config:
    WRAPPER_CODE_PATH = "../wrapper"
    INCLUDE_PATH = "../../include"

    LAYOUT_TAG_SET = [0, 1]  # 0 is RowMajor, 1 is ColumnMajor
    LAYOUT_TAG_MAP = {0: "Catlass::layout::RowMajor", 1: "Catlass::layout::ColumnMajor"}

    DTYPE_MAP = {"half": 0, "float": 1}

    @staticmethod
    def get_tiling_key(
        kernel_serial, dtype, l_tag_a, l_tag_b, l_tag_c, p_tag_a, p_tag_b, p_tag_c
    ):
        part1 = kernel_serial  # 56-63
        part2 = Config.DTYPE_MAP[dtype] << 4  # 48-55
        part3 = 0  # 40-47
        part4 = 0  # 32-39
        part5 = 0  # 24-31
        part6 = (p_tag_a << 4) | p_tag_b  # 16-23
        part7 = (p_tag_c << 4) | l_tag_a  # 8-15
        part8 = (l_tag_b << 4) | l_tag_c  # 0-7
        hex_str = f"0x{part1:02x}{part2:02x}{part3:02x}{part4:02x}{part5:02x}{part6:02x}{part7:02x}{part8:02x}"
        return hex_str
