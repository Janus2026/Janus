#ifndef INCLUDE_ITERTOR_H
#define INCLUDE_ITERTOR_H

#include "common_func.h"
#include "hardware.h"
#include "kernel_operator.h"
#include "layout.h"
#include "mem.h"

/////////////////////////////////////////////////////
// gm_to_l1
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DataType, DataFormat FormatInGM, DataFormat FormatInL1>
struct gm_to_l1 {
    __aicore__ gm_to_l1(AscendC::LocalTensor<DataType> l1Tensor,
                        AscendC::GlobalTensor<DataType> gmTensor,
                        uint32_t nTileActual,
                        uint32_t nTileCeil,
                        uint32_t nVal,
                        uint32_t dTileActual,
                        uint32_t dTileCeil,
                        uint32_t dVal) {};
};

/////////////////////////////////////////////////////
// l1_to_l0_a
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DataType, bool IsTransPose, DataFormat DFmtIn, DataFormat DFmtOut>
struct l1_to_l0_a {
    __aicore__ l1_to_l0_a(AscendC::LocalTensor<DataType> l0Tensor,
                          AscendC::LocalTensor<DataType> l1Tensor,
                          uint32_t mTileCeil,
                          uint32_t kPartCeil,
                          uint32_t mSrcStride,
                          uint32_t kSrcStride,
                          uint32_t mDstStride,
                          uint32_t kDstStride) {};
};

/////////////////////////////////////////////////////
// l1_to_l0_b
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DataType, bool IsTransPose, DataFormat DFmtIn, DataFormat DFmtOut>
struct l1_to_l0_b {
    __aicore__ l1_to_l0_b(AscendC::LocalTensor<DataType> l0Tensor,
                          AscendC::LocalTensor<DataType> l1Tensor,
                          uint32_t nTileCeil,
                          uint32_t kPartCeil,
                          uint32_t nSrcStride,
                          uint32_t kSrcStride,
                          uint32_t nDstStride,
                          uint32_t kDstStride) {};
};

// l1_to_l0_a
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DataType, bool IsTransPose, bool IsVectore>
struct l1_to_l0_a_v1 {
    __aicore__ l1_to_l0_a_v1(AscendC::LocalTensor<DataType> l0_tensor,
                             AscendC::LocalTensor<DataType> l1_tensor,
                             uint32_t m_tile_ceil,
                             uint32_t k_tile_ceil,
                             uint32_t k_part,
                             uint32_t k_part_ceil,
                             uint32_t k_part_idx) {};
};

/////////////////////////////////////////////////////
// l1_to_l0_b
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DataType, bool IsTransPose, bool IsVectore>
struct l1_to_l0_b_v1 {
    __aicore__ l1_to_l0_b_v1(AscendC::LocalTensor<DataType> l0_tensor,
                             AscendC::LocalTensor<DataType> l1_tensor,
                             int32_t n_tile_ceil,
                             int32_t k_tile_ceil,
                             int32_t k_part_ceil,
                             int32_t k_part_idx) {};
};

/////////////////////////////////////////////////////
// l0c_to_gm
/////////////////////////////////////////////////////
template <ArchType ArchTag, DataFormat OutFormatType, typename OutDataType, typename L0CDataType>
struct l0c_to_gm {
    __aicore__ l0c_to_gm(AscendC::GlobalTensor<OutDataType> gmTensor,
                         AscendC::LocalTensor<L0CDataType> l0cTensor,
                         uint32_t mTileActual,
                         uint32_t nTileActual,
                         uint32_t mTileCeil,
                         uint32_t nActual) {};
};

/////////////////////////////////////////////////////
// l0c_to_l1
/////////////////////////////////////////////////////
template <ArchType ArchTag, DataFormat LayoutOut, typename ElementOut, typename ElementIn>
struct l0c_to_l1 {
    __aicore__ l0c_to_l1(AscendC::LocalTensor<ElementOut> l1Tensor,
                         AscendC::LocalTensor<ElementIn> l0cTensor,
                         AscendC::LocalTensor<uint64_t> deqTensor,
                         uint32_t mTileActual,
                         uint32_t nTileActual,
                         uint32_t mTileCeil,
                         uint32_t nActual) {};
};

#include "iterators/gm_to_l1_iterator.inc"
#include "iterators/gm_to_ub_iterator.inc"
#include "iterators/l0c_to_gm_iterator.inc"
#include "iterators/l0c_to_l1_iterator.inc"
#include "iterators/l0c_to_ub_iterator.inc"
#include "iterators/l1_to_bt_iterator.inc"
#include "iterators/l1_to_fb_iterator.inc"
#include "iterators/l1_to_l0_iterator.inc"
#include "iterators/l1_to_ub_iterator.inc"
#endif
