#include "../iterator.h"

/////////////////////////////////////////////////////
// l1_to_ub
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DataType>
struct l1_to_ub {
    __aicore__ l1_to_ub(AscendC::LocalTensor<DataType> ubTensor,
                        AscendC::LocalTensor<DataType> l1Tensor,
                        uint16_t nBurst,
                        uint16_t lenBurst,
                        uint16_t srcStride,
                        uint16_t dstStride)
    {
        AscendC::DataCopy(ubTensor, l1Tensor, AscendC::DataCopyParams(nBurst, lenBurst, srcStride, dstStride));
    };
};

/////////////////////////////////////////////////////
// ub_to_l1
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DataType>
struct ub_to_l1 {
    __aicore__ ub_to_l1(AscendC::LocalTensor<DataType> l1Tensor,
                        AscendC::LocalTensor<DataType> ubTensor,
                        uint16_t nBurst,
                        uint16_t lenBurst,
                        uint16_t srcStride,
                        uint16_t dstStride)
    {
        AscendC::DataCopy(l1Tensor, ubTensor, AscendC::DataCopyParams(nBurst, lenBurst, srcStride, dstStride));
    };
};