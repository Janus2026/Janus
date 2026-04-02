#include "../iterator.h"

/////////////////////////////////////////////////////
// l1_to_bt
/////////////////////////////////////////////////////

// Partial specialization for V220
template <ArchType ArchTag, typename DataType>
struct l1_to_bt {
    using HardwareParams = HardwareInfo<ArchTag>;
    static constexpr uint32_t BLOCK_SIZE = HardwareParams::btBlockSize / sizeof(DataType);

    __aicore__ l1_to_bt(AscendC::LocalTensor<DataType> biasTableTensor,
                        AscendC::LocalTensor<DataType> biasL1Tensor,
                        uint32_t ntileActual)
    {
        AscendC::DataCopy(
            biasTableTensor, biasL1Tensor, {1, static_cast<uint16_t>(CeilDiv<BLOCK_SIZE>(ntileActual)), 0, 0});
    };
};