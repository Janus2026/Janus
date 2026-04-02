#include "../iterator.h"

/////////////////////////////////////////////////////
// l1_to_fb
/////////////////////////////////////////////////////

// Partial specialization for V220
template <ArchType ArchTag, typename DataType>
struct l1_to_fb {
    using HardwareParams = HardwareInfo<ArchTag>;
    static constexpr uint32_t BLOCK_SIZE = HardwareParams::fbBlockSize / sizeof(DataType);

    __aicore__
    l1_to_fb(AscendC::LocalTensor<DataType> fbTensor, AscendC::LocalTensor<DataType> l1Tensor, uint32_t ntileActual)
    {
        copy_cbuf_to_fbuf((__fbuf__ DataType *)fbTensor.GetPhyAddr(),
                          (__cbuf__ DataType *)l1Tensor.GetPhyAddr(),
                          1,
                          CeilDiv<BLOCK_SIZE>(ntileActual),
                          0,
                          0);
    };
};