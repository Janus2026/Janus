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

import logging
from itertools import product
from dataclasses import dataclass

import library
from gemm_operation import GemmOperation
from manifest import OperationRegistry

LOGGER = logging.getLogger(__name__)


@dataclass
class SearchSpaceConfiguration:
    kernel_type: str # e.g. 'basic_matmul'

    data_type_a: library.DataType # e.g. library.DataType.fp16/fp32
    data_type_b: library.DataType
    data_type_c: library.DataType

    layout_a: library.LayoutType # e.g. library.LayoutType.RowMajor/ColumnMajor
    layout_b: library.LayoutType
    layout_c: library.LayoutType

    l1_tile_m_range: tuple # (min, max), e.g. (32, 128)
    l1_tile_n_range: tuple
    l1_tile_k_range: tuple

    block_swizzle: str # e.g. 'Gemm::Block::GemmIdentityBlockSwizzle<3, 0>'


def generate_tile_shape_default(
    l1_tile_m_range: tuple,
    l1_tile_n_range: tuple,
    l1_tile_k_range: tuple,
):
    l0_tile_m_range = l1_tile_m_range
    l0_tile_n_range = l1_tile_n_range
    l0_tile_k_range = tuple(int(x / 4) for x in l1_tile_k_range)

    tile_shapes = list(generate_tile_shapes(
        tile_shape_constraint_for_pingpong, # 默认减枝函数
        element_sizes=(2, 2, 4), # size of ElementA, ElementB, ElementAccumulator
        stages=(2),
        step=16,
        tile_shape_range=TileShapeRange(
            l1_tile_m_range=l1_tile_m_range,
            l1_tile_n_range=l1_tile_n_range,
            l1_tile_k_range=l1_tile_k_range,
            l0_tile_m_range=l0_tile_m_range,
            l0_tile_n_range=l0_tile_n_range,
            l0_tile_k_range=l0_tile_k_range,
        )
    ))
    return tile_shapes


def register_custom_kernel(
    config: SearchSpaceConfiguration,
    manifest
):

    tile_shapes = generate_tile_shape_default(
        config.l1_tile_m_range, config.l1_tile_n_range, config.l1_tile_k_range
    )

    LOGGER.info(f'{config.kernel_type} tile_shapes size={len(tile_shapes)}')

    for tile_shape in tile_shapes:
        l1_tile_shape, l0_tile_shape = tile_shape
        tensor_a = library.GemmTypeDescription(config.data_type_a, config.layout_a)
        tensor_b = library.GemmTypeDescription(config.data_type_b, config.layout_b)
        tensor_c = library.GemmTypeDescription(config.data_type_c, config.layout_c)
        op = GemmOperation(
            kernel_type=config.kernel_type,
            l1_tile_shape=l1_tile_shape,
            l0_tile_shape=l0_tile_shape,
            a_type=tensor_a,
            b_type=tensor_b,
            c_type=tensor_c,
            block_swizzle=config.block_swizzle,
        )
        manifest.append(op)


############### search space generation methods ###############
L1_SIZE_MAX = 512 * 1024
L0A_SIZE_MAX = 64 * 1024
L0B_SIZE_MAX = 64 * 1024
L0C_SIZE_MAX = 128 * 1024


def tile_shape_constraint_for_pingpong(
    l1_tile_shape,
    l0_tile_shape,
    element_sizes_tuple,
    stages_tuple
):
    # constraint function for "Gemm::MmadAtlasA2Pingpong"
    l1_m, l1_n, l1_k = l1_tile_shape
    l0_m, l0_n, l0_k = l0_tile_shape
    element_a_size, element_b_size, element_accumulator_size = element_sizes_tuple
    stages = stages_tuple

    l1a_tile_size = l1_m * l1_k * element_a_size
    l1b_tile_size = l1_n * l1_k * element_b_size
    l0a_tile_size = l0_m * l0_k * element_a_size
    l0b_tile_size = l0_k * l0_n * element_b_size
    l0c_tile_size = l0_m * l0_n * element_accumulator_size

    # the basic blocks of L1 and L0 differ on the m and n axes is not supported yet
    if l1_m != l0_m or l1_n != l0_n:
        return False

    # check L1
    if (l1a_tile_size * stages + l1b_tile_size * stages) > L1_SIZE_MAX:
        return False

    # check L0A
    if l0a_tile_size * stages > L0A_SIZE_MAX:
        return False

    # check L0B
    if l0b_tile_size * stages > L0B_SIZE_MAX:
        return False

    # check L0C
    if l0c_tile_size > L0C_SIZE_MAX:
        return False

    return True


def tile_shape_constraint_for_preload_async(
    l1_tile_shape,
    l0_tile_shape,
    element_sizes_tuple,
    stages_tuple
):
    # constraint function for "Gemm::MmadAtlasA2PreloadAsync"
    l1_m, l1_n, l1_k = l1_tile_shape
    l0_m, l0_n, l0_k = l0_tile_shape
    element_a_size, element_b_size, element_accumulator_size = element_sizes_tuple
    _, l1_stages, l0a_stages, l0b_stages, l0c_stages, = stages_tuple

    l1a_tile_size = l1_m * l1_k * element_a_size
    l1b_tile_size = l1_n * l1_k * element_b_size
    l0a_tile_size = l0_m * l0_k * element_a_size
    l0b_tile_size = l0_k * l0_n * element_b_size
    l0c_tile_size = l0_m * l0_n * element_accumulator_size

    # the basic blocks of L1 and L0 differ on the m and n axes is not supported yet
    if l1_m != l0_m or l1_n != l0_n:
        return False

    # check L1
    if (l1a_tile_size * l1_stages + l1b_tile_size * l1_stages) > L1_SIZE_MAX:
        return False

    # check L0A
    if l0a_tile_size * l0a_stages > L0A_SIZE_MAX:
        return False

    # check L0B
    if l0b_tile_size * l0b_stages > L0B_SIZE_MAX:
        return False

    # check L0C
    if l0c_tile_size * l0c_stages > L0C_SIZE_MAX:
        return False

    return True


@dataclass
class TileShapeRange:
    l1_tile_m_range: tuple
    l1_tile_n_range: tuple
    l1_tile_k_range: tuple
    l0_tile_m_range: tuple
    l0_tile_n_range: tuple
    l0_tile_k_range: tuple


def generate_tile_shapes(
    constraint_func: callable = tile_shape_constraint_for_pingpong,
    element_sizes: tuple = (2, 2, 4),
    stages: tuple = (2),
    step: int = 16,
    tile_shape_range: TileShapeRange = TileShapeRange(
        l1_tile_m_range=(32, 128),
        l1_tile_n_range=(128, 256),
        l1_tile_k_range=(128, 256),
        l0_tile_m_range=(32, 128),
        l0_tile_n_range=(128, 256),
        l0_tile_k_range=(32, 64)
    )
):
    if step % 16 != 0:
        raise ValueError(f"step must be multiples of 16")

    def generator(
        element_sizes,
        stages
    ):
        params_ranges = [
            range(tile_shape_range.l1_tile_m_range[0], tile_shape_range.l1_tile_m_range[1] + step, step),
            range(tile_shape_range.l1_tile_n_range[0], tile_shape_range.l1_tile_n_range[1] + step, step),
            range(tile_shape_range.l1_tile_k_range[0], tile_shape_range.l1_tile_k_range[1] + step, step),
            range(tile_shape_range.l0_tile_m_range[0], tile_shape_range.l0_tile_m_range[1] + step, step),
            range(tile_shape_range.l0_tile_n_range[0], tile_shape_range.l0_tile_n_range[1] + step, step),
            range(tile_shape_range.l0_tile_k_range[0], tile_shape_range.l0_tile_k_range[1] + step, step)
        ]
        for l1_m, l1_n, l1_k, l0_m, l0_n, l0_k in product(*params_ranges):
            if constraint_func is None or constraint_func(
                (l1_m, l1_n, l1_k),
                (l0_m, l0_n, l0_k),
                element_sizes,
                stages
            ):
                yield ((l1_m, l1_n, l1_k), (l0_m, l0_n, l0_k))
    return generator(element_sizes, stages)

############### search space generation methods end ###############


################## basic_matmul ##################
@OperationRegistry.register('basic_matmul')
def register_gemm_basic_matmul_operation(manifest):

    layouts = [
        [library.LayoutType.RowMajor, library.LayoutType.RowMajor, library.LayoutType.RowMajor],
    ]

    data_types = [
        [library.DataType.fp16, library.DataType.fp16, library.DataType.fp16]
    ]

    # 设定L1/L0TileShape的搜索范围、搜索步长、减枝函数，生成范围内全量搜索结点
    tile_shapes = list(generate_tile_shapes(
        tile_shape_constraint_for_pingpong, # 自定义减枝函数
        element_sizes=(2, 2, 4), # size of ElementA, ElementB, ElementAccumulator
        stages=(2),
        step=16,
        tile_shape_range=TileShapeRange(
            l1_tile_m_range=(32, 128),
            l1_tile_n_range=(128, 256),
            l1_tile_k_range=(128, 256),
            l0_tile_m_range=(32, 128),
            l0_tile_n_range=(128, 256),
            l0_tile_k_range=(32, 64)
        )
    ))
    LOGGER.info(f'basic_matmul tile_shapes size={len(tile_shapes)}')

    block_swizzle_descriptions = [
        'Gemm::Block::GemmIdentityBlockSwizzle<3, 0>',
        # 可自定义其他Swizzle参数
    ]

    # 正交tiling参数组合
    for layout, data_type, tile_shape, block_swizzle in product(
        layouts, data_types, tile_shapes, block_swizzle_descriptions
    ):
        l1_tile_shape, l0_tile_shape = tile_shape
        tensor_a = library.GemmTypeDescription(data_type[0], layout[0])
        tensor_b = library.GemmTypeDescription(data_type[1], layout[1])
        tensor_c = library.GemmTypeDescription(data_type[2], layout[2])
        op = GemmOperation(
            kernel_type='basic_matmul',
            l1_tile_shape=l1_tile_shape,
            l0_tile_shape=l0_tile_shape,
            a_type=tensor_a,
            b_type=tensor_b,
            c_type=tensor_c,
            block_swizzle=block_swizzle,
        )
        manifest.append(op)
################## basic_matmul end ##################


################## grouped_matmul ##################
@OperationRegistry.register('grouped_matmul')
def register_gemm_grouped_matmul_operation(manifest):

    layouts = [
        [library.LayoutType.ColumnMajor, library.LayoutType.RowMajor, library.LayoutType.RowMajor],
    ]
    data_types = [
        [library.DataType.fp16, library.DataType.fp16, library.DataType.fp16],
    ]
    block_swizzle_descriptions = [
        'Gemm::Block::GemmIdentityBlockSwizzle<3, 1>',
    ]

    # generate L1/L0TileShape search space
    tile_shapes = list(generate_tile_shapes(
        tile_shape_constraint_for_preload_async, # 自定义减枝函数
        element_sizes=(2, 2, 4), # size of ElementA, ElementB, ElementAccumulator
        stages=(1, 2, 4, 2, 1), # Preload/L1/L0A/L0B/L0C stages
        step=16,
        tile_shape_range=TileShapeRange(
            l1_tile_m_range=(128, 256),
            l1_tile_n_range=(128, 256),
            l1_tile_k_range=(128, 256),
            l0_tile_m_range=(128, 256),
            l0_tile_n_range=(128, 256),
            l0_tile_k_range=(32, 64)
        )
    ))
    LOGGER.info(f'grouped_matmul tile_shapes size={len(tile_shapes)}')

    # 正交tiling参数组合
    for layout, data_type, tile_shape, block_swizzle in product(
        layouts, data_types, tile_shapes, block_swizzle_descriptions
    ):
        l1_tile_shape, l0_tile_shape = tile_shape
        tensor_a = library.GemmTypeDescription(data_type[0], layout[0])
        tensor_b = library.GemmTypeDescription(data_type[1], layout[1])
        tensor_c = library.GemmTypeDescription(data_type[2], layout[2])
        op = GemmOperation(
            kernel_type='grouped_matmul',
            l1_tile_shape=l1_tile_shape,
            l0_tile_shape=l0_tile_shape,
            a_type=tensor_a,
            b_type=tensor_b,
            c_type=tensor_c,
            block_swizzle=block_swizzle,
        )
        manifest.append(op)
################## grouped_matmul end ##################