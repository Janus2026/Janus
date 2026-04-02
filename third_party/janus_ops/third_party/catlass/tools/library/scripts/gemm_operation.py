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
import re
import library
from utils import KernelGroupFile


class GemmOperation:
    def __init__(
        self,
        kernel_type: str,
        l1_tile_shape: list,
        l0_tile_shape: list,
        a_type: library.GemmTypeDescription,
        b_type: library.GemmTypeDescription,
        c_type: library.GemmTypeDescription,
        block_swizzle: str,
        arch: library.ArchTag = library.ArchTag.A2,
    ):
        self.operation_type = 'gemm'
        self.kernel_type = kernel_type
        self.l1_tile_shape = l1_tile_shape
        self.l0_tile_shape = l0_tile_shape
        self.a_type = a_type
        self.b_type = b_type
        self.c_type = c_type
        self.block_swizzle = block_swizzle
        self.arch = arch

        self.kernel_name = self.get_name()

        self.kernel_instance_generators = {
            'basic_matmul': BasicMatmulKernelInstance,
            'grouped_matmul': GroupedMatmulKernelInstance,
        }

        self.body_template = """
void Register_{kernel_name}(Manifest &manifest)
{{
    using {kernel_name} =
        {kernel_instance};

    manifest.Append(
        new {cpp_instance}<{kernel_name}>(
            "{kernel_name}"
        )
    );
}}
"""

    def get_name(self):

        template = (
            "catlass_{operation_type}_{kernel_type}_"
            "{data_type_a}x{layout_a}_"
            "{data_type_b}x{layout_b}_"
            "{data_type_c}x{layout_c}_"
            "{l1_tile_shape}_"
            "{l0_tile_shape}_"
            "{block_swizzle}"
        )

        return template.format(
            operation_type=self.operation_type,
            kernel_type=self.kernel_type,
            data_type_a=self.a_type.element_type.get_name(),
            data_type_b=self.b_type.element_type.get_name(),
            data_type_c=self.c_type.element_type.get_name(),
            layout_a=self.a_type.layout.get_name(),
            layout_b=self.b_type.layout.get_name(),
            layout_c=self.c_type.layout.get_name(),
            l1_tile_shape='x'.join(str(val) for val in self.l1_tile_shape),
            l0_tile_shape='x'.join(str(val) for val in self.l0_tile_shape),
            block_swizzle=self.get_block_swizzle_name()
        )

    def get_block_swizzle_name(self):
        match = re.search(r'<(\d+)\s*,\s*(\d+)\s*>', self.block_swizzle)
        if not match:
            return ''
        num1 = match.group(1)
        num2 = match.group(2)
        return f'swizzle{num1}x{num2}'

    def generate_src(self):
        if self.kernel_type in self.kernel_instance_generators:
            instance_geneorator = self.kernel_instance_generators[self.kernel_type]()
        else:
            raise Exception(f'no kernel instance registered for {self.kernel_type}')
        kernel_instance_src = instance_geneorator.gen_src(self)

        body_src = self.body_template.format(
            kernel_name=self.kernel_name,
            kernel_instance=kernel_instance_src,
            cpp_instance=instance_geneorator.cpp_instance,
        )

        return instance_geneorator.custom_headers, instance_geneorator.custom_common_decls, body_src


class GemmOperationGenerator:
    def __init__(self, operation_type, generated_dir):
        self.generated_dir = generated_dir
        self.operation_type = operation_type
        self.kernel_names = []
        self.kernel_instances = []

        # critical: avoid creating too many files that bisheng-compiler cannot not handle
        self.kernel_group_files = []
        self.curr_file_id = 0

        self.function_decl_template = """void Register_{kernel_name}(Manifest &manifest);\n"""
        self.function_call_template = """    Register_{kernel_name}(manifest);\n"""

        self.register_template = """
#include "catlass/library/operation.h"
#include "catlass/library/manifest.h"

namespace Catlass {{
namespace Library {{

{function_decls}

void RegisterCatlass{operation_type}Operations(Manifest &manifest)
{{
{function_calls}
}}

}}
}}
"""

        self.gemm_headers = """
#include "catlass/library/operation.h"
#include "catlass/library/manifest.h"

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/device/device_gemm.hpp"

#include "gemm_operation.h"
"""

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        for file in self.kernel_group_files:
            save_dir = os.path.join(self.generated_dir, self.operation_type)
            file.write_in_dir(save_dir)

    def gen(self, name, operation):
        kernel_name = name
        self.kernel_names.append(kernel_name)

        headers, decls, body = operation.generate_src()
        file = self.get_next_kernel_group_file()
        file.add_instance(headers, decls, body)

    def get_next_kernel_group_file(self):
        GROUP_FILE_NUM = 64
        if len(self.kernel_group_files) < GROUP_FILE_NUM:
            file = KernelGroupFile(f'catlass_{self.operation_type}_kernel_group_{self.curr_file_id}.cpp')
            file.add_headers(self.gemm_headers)
            self.kernel_group_files.append(file)
            self.curr_file_id = (self.curr_file_id + 1) % GROUP_FILE_NUM
            return self.kernel_group_files[-1]

        file = self.kernel_group_files[self.curr_file_id]
        self.curr_file_id = (self.curr_file_id + 1) % GROUP_FILE_NUM
        return file


class BasicMatmulKernelInstance:
    def __init__(self):
        self.cpp_instance = 'BasicMatmulGemmOperation'
        self.custom_headers = '#include "catlass/gemm/kernel/basic_matmul.hpp"'
        self.custom_common_decls = ''
        self.template = """
        Gemm::Device::DeviceGemm<
            Gemm::Kernel::BasicMatmul<
                Gemm::Block::BlockMmad<
                    Gemm::MmadAtlasA2Pingpong<true>,
                    GemmShape<{l1_m}, {l1_n}, {l1_k}>,
                    GemmShape<{l0_m}, {l0_n}, {l0_k}>,
                    Gemm::GemmType<{element_a}, {layout_a}>,
                    Gemm::GemmType<{element_b}, {layout_b}>,
                    Gemm::GemmType<{element_c}, {layout_c}>
                >,
                void,
                {block_swizzle}
            >
        >"""


    def gen_src(self, gemm_operation):
        src = self.template.format(
            l1_m=str(gemm_operation.l1_tile_shape[0]),
            l1_n=str(gemm_operation.l1_tile_shape[1]),
            l1_k=str(gemm_operation.l1_tile_shape[2]),
            l0_m=str(gemm_operation.l0_tile_shape[0]),
            l0_n=str(gemm_operation.l0_tile_shape[1]),
            l0_k=str(gemm_operation.l0_tile_shape[2]),
            element_a=gemm_operation.a_type.element_type.to_code(),
            element_b=gemm_operation.b_type.element_type.to_code(),
            element_c=gemm_operation.c_type.element_type.to_code(),
            layout_a=gemm_operation.a_type.layout.to_code(),
            layout_b=gemm_operation.b_type.layout.to_code(),
            layout_c=gemm_operation.c_type.layout.to_code(),
            block_swizzle=gemm_operation.block_swizzle
        )
        return src


class GroupedMatmulKernelInstance:
    def __init__(self):
        self.cpp_instance = 'GroupedMatmulGemmOperation'
        self.custom_headers = '#include "catlass/gemm/kernel/grouped_matmul.hpp"'
        self.custom_common_decls = ''
        self.template = """
        Gemm::Device::DeviceGemm<
            Gemm::Kernel::GroupedMatmul<
                Gemm::Block::BlockMmad<
                    Gemm::MmadAtlasA2PreloadAsync<1,2,4,2,1,true,true>,
                    GemmShape<{l1_m}, {l1_n}, {l1_k}>,
                    GemmShape<{l0_m}, {l0_n}, {l0_k}>,
                    Gemm::GemmType<{element_a}, {layout_a}>,
                    Gemm::GemmType<{element_b}, {layout_b}>,
                    Gemm::GemmType<{element_c}, {layout_c}>
                >,
                void,
                {block_swizzle}
            >
        >"""


    def gen_src(self, gemm_operation):
        src = self.template.format(
            l1_m=str(gemm_operation.l1_tile_shape[0]),
            l1_n=str(gemm_operation.l1_tile_shape[1]),
            l1_k=str(gemm_operation.l1_tile_shape[2]),
            l0_m=str(gemm_operation.l0_tile_shape[0]),
            l0_n=str(gemm_operation.l0_tile_shape[1]),
            l0_k=str(gemm_operation.l0_tile_shape[2]),
            element_a=gemm_operation.a_type.element_type.to_code(),
            element_b=gemm_operation.b_type.element_type.to_code(),
            element_c=gemm_operation.c_type.element_type.to_code(),
            layout_a=gemm_operation.a_type.layout.to_code(),
            layout_b=gemm_operation.b_type.layout.to_code(),
            layout_c=gemm_operation.c_type.layout.to_code(),
            block_swizzle=gemm_operation.block_swizzle
        )
        return src

