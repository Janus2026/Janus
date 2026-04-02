/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <sstream>
#include <cstring>
#include <securec.h>
#include "atb_speed/log.h"
#include "atb_speed/utils/check_util.h"
#include "utils.h"

namespace atb_speed {
namespace common {

atb::SVector<int64_t> GetCopyTensorStride(atb::Dims &tensorDims)
{
    atb::SVector<int64_t> tmpStrides(tensorDims.dimNum, 1);
    if (tensorDims.dimNum > 8) {  // 8: tensortranslated
        ATB_SPEED_LOG_ERROR("Tensor's dimNum is larger than 8, `GetCopyTensorStride` failed.");
        return tmpStrides;
    }
    for (int64_t i = static_cast<int64_t>(tensorDims.dimNum) - 2; i >= 0; i--) {
        tmpStrides[i] = CheckIntMulOverFlow(tensorDims.dims[i + 1], tmpStrides[i + 1]);
    }
    return tmpStrides;
}

atb::SVector<int64_t> GetTransposeTensorStride(atb::Dims &tensorDims)
{
    atb::SVector<int64_t> tmptransposeStrides(tensorDims.dimNum, 1);
    tmptransposeStrides[tensorDims.dimNum - 1] = tensorDims.dims[tensorDims.dimNum - 1];
    if (tensorDims.dimNum == 3) {  // 3: translated
        tmptransposeStrides[0] = CheckIntMulOverFlow(  // 0: translated0translated
            tensorDims.dims[1], tensorDims.dims[2]);  // 1, 2: translated1translated2translated
    }
    return tmptransposeStrides;
}

atb::Status CallAclCreateTensor(atb::Dims &viewDims, atb::Dims &storageDims, atb::Tensor &atbTensor,
    std::shared_ptr<AclNNTensor> aclnnTensor)
{
    aclnnTensor->tensor = aclCreateTensor(viewDims.dims,
        viewDims.dimNum,
        atbTensor.desc.dtype,
        aclnnTensor->strides.data(),
        0,
        atbTensor.desc.format,
        storageDims.dims,
        storageDims.dimNum,
        atbTensor.deviceData);
    if (aclnnTensor->tensor == nullptr) {
        return atb::ERROR_INTERNAL_ERROR;
    }
    return atb::NO_ERROR;
}

bool IsA2()
{
    // translatedatbtranslated:atbtranslated
    const uint32_t lenOfAtlasA2 = 10;
    std::string socName = aclrtGetSocName();
    ATB_SPEED_LOG_DEBUG("SocVersionName:" << std::string(socName));
    bool isA2 = (std::string(socName).find("Ascend910B") != std::string::npos &&
        std::string(socName).length() > lenOfAtlasA2) ||
        std::string(socName).find("Ascend910_93") != std::string::npos;
    return isA2;
}

bool IsA3()
{
    std::string socName = aclrtGetSocName();
    ATB_SPEED_LOG_DEBUG("SocVersionName:" << std::string(socName));
    bool isA3 = std::string(socName).find("Ascend910_93") != std::string::npos;
    return isA3;
}

bool Is310P()
{
    std::string socName = aclrtGetSocName();
    ATB_SPEED_LOG_DEBUG("SocVersionName:" << std::string(socName));
    bool is310P = std::string(socName).find("Ascend310P") != std::string::npos;
    return is310P;
}

atb::Tensor SqueezeBatchSeq(atb::Tensor atbTensor)
{
    if (atbTensor.desc.shape.dimNum == DIM3) {
        atbTensor.desc.shape.dimNum = DIM2;
        atbTensor.desc.shape.dims[DIM0] = CheckIntMulOverFlow(
            atbTensor.desc.shape.dims[DIM0], atbTensor.desc.shape.dims[DIM1]);
        atbTensor.desc.shape.dims[DIM1] = atbTensor.desc.shape.dims[DIM2];
    }
    return atbTensor;
}

std::string PrintAclNNVariankPack(const AclNNVariantPack &aclnnVariantPack)
{
    std::stringstream ss;
    ss << "Plugin Op Cache: AclNNVariantPack ";
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); i++) {
        const atb::TensorDesc &tensorDesc = aclnnVariantPack.aclInTensors[i]->atbTensor.desc;
        ss << "index " << i << " dtype " << tensorDesc.dtype
           << " format " << tensorDesc.format << " dimNum " << tensorDesc.shape.dimNum;
        for (uint64_t j = 0; j < std::min(tensorDesc.shape.dimNum, static_cast<uint64_t>(8)); j++) {  // 8: tensortranslated
            ss << "dim[" << j << "]=" << tensorDesc.shape.dims[j] << " ";
        }
    }
    return ss.str();
}

std::string PrintATBVariankPack(const atb::VariantPack &atbVariantPack)
{
    std::stringstream ss;
    ss << "Plugin Op Cache: ATBVariantPack ";
    for (size_t i = 0; i < atbVariantPack.inTensors.size(); i++) {
        const atb::TensorDesc &tensorDesc = atbVariantPack.inTensors[i].desc;
        ss << "index " << i << " dtype " << tensorDesc.dtype
           << " format " << tensorDesc.format << " dimNum " << tensorDesc.shape.dimNum;
        for (uint64_t j = 0; j < std::min(tensorDesc.shape.dimNum, static_cast<uint64_t>(8)); j++) {  // 8: tensortranslated
            ss << "dim[" << j << "]=" << tensorDesc.shape.dims[j] << " ";
        }
    }
    return ss.str();
}

bool IsHostDataEqual(const std::shared_ptr<AclNNTensor> tensorA, const atb::Tensor &tensorB, int tensorIdx)
{
    if (tensorA->intArrayHostData.intArray != nullptr && tensorB.hostData == nullptr) {
        ATB_SPEED_LOG_DEBUG("Plugin Op Cache: tensor index " << tensorIdx
                        << " aclnnVariantPack hostData is not null but atbVariantPack hostData is");
        return false;
    }
    if (tensorA->intArrayHostData.intArray == nullptr && tensorB.hostData != nullptr) {
        ATB_SPEED_LOG_DEBUG("Plugin Op Cache: tensor index " << tensorIdx
                        << " aclnnVariantPack hostData is null but atbVariantPack hostData is not");
        return false;
    }
    if (tensorA->intArrayHostData.intArray != nullptr && tensorB.hostData != nullptr) {
        if (tensorA->intArrayHostData.dataOri.size() * 4 != tensorB.dataSize) {  // 8: int64_t in bytes
            ATB_SPEED_LOG_DEBUG("Plugin Op Cache: tensor index " << tensorIdx << " dataSize not equal");
            return false;
        }
        if (memcmp(tensorA->intArrayHostData.dataOri.data(), tensorB.hostData, tensorB.dataSize) != 0) {
            ATB_SPEED_LOG_DEBUG("Plugin Op Cache: tensor index " << tensorIdx << " hostData not equal");
            return false;
        }
    }
    return true;
}

bool IsTensorDescEqual(const atb::TensorDesc &tensorDescA, const atb::TensorDesc &tensorDescB, int tensorIdx)
{
    if (tensorDescA.dtype != tensorDescB.dtype) {
        ATB_SPEED_LOG_DEBUG("Plugin Op Cache: tensor index " << tensorIdx
                        << " dtype not equal, aclnnVariantPack dtype " << tensorDescA.dtype
                        << " atbVariantPack dtype " << tensorDescB.dtype);
        return false;
    }
    if (tensorDescA.format != tensorDescB.format) {
        ATB_SPEED_LOG_DEBUG("Plugin Op Cache: tensor index " << tensorIdx
                        << " format not equal, aclnnVariantPack format " << tensorDescA.format
                        << " atbVariantPack format " << tensorDescB.format);
        return false;
    }
    if (tensorDescA.shape.dimNum != tensorDescB.shape.dimNum || \
        tensorDescA.shape.dimNum > 8 || tensorDescA.shape.dimNum <= 0) {  // 8: tensortranslated
        ATB_SPEED_LOG_DEBUG("Plugin Op Cache: tensor index " << tensorIdx
                        << " dimNum not equal, aclnnVariantPack dimNum " << tensorDescA.shape.dimNum
                        << " atbVariantPack dimNum " << tensorDescB.shape.dimNum);
        return false;
    }
    for (uint64_t j = 0; j < tensorDescA.shape.dimNum; j++) {
        if (tensorDescA.shape.dims[j] != tensorDescB.shape.dims[j]) {
            ATB_SPEED_LOG_DEBUG("Plugin Op Cache: : tensor index " << tensorIdx
                            << " shape.dims " << j << " not equal, aclnnVariantPack value "
                            << tensorDescA.shape.dims[j] << " atbVariantPack value " << tensorDescB.shape.dims[j]);
            return false;
        }
    }
    return true;
}

bool AreTensorVectorsEqual(
    const atb::SVector<std::shared_ptr<AclNNTensor>> &aclnnTensors, const atb::SVector<atb::Tensor> &atbTensors)
{
    // Check the size of two vectors
    if (aclnnTensors.size() != atbTensors.size()) {
        ATB_SPEED_LOG_DEBUG("Plugin Op Cache: size not equal, aclnnVariantPack size "
                      << aclnnTensors.size() << " atbVariantPack size "
                      << atbTensors.size());
        return false;
    }

    // Check if every tensor in each vector has consistent data type, format, shape and host data.
    for (size_t i = 0; i < aclnnTensors.size(); i++) {
        const std::shared_ptr<AclNNTensor> tensorA = aclnnTensors[i];
        const atb::Tensor &tensorB = atbTensors[i];

        if (!IsHostDataEqual(tensorA, tensorB, i)) {
            return false;
        }

        if (!IsTensorDescEqual(tensorA->atbTensor.desc, tensorB.desc, i)) {
            return false;
        }
    }

    return true;
}

bool IsVariankPackEqual(const AclNNVariantPack &aclnnVariantPack, const atb::VariantPack &atbVariantPack)
{
    ATB_SPEED_LOG_DEBUG(PrintAclNNVariankPack(aclnnVariantPack));
    ATB_SPEED_LOG_DEBUG(PrintATBVariankPack(atbVariantPack));

    if (!AreTensorVectorsEqual(aclnnVariantPack.aclInTensors, atbVariantPack.inTensors)) {
        return false;
    }

    if (!AreTensorVectorsEqual(aclnnVariantPack.aclOutTensors, atbVariantPack.outTensors)) {
        return false;
    }

    ATB_SPEED_LOG_DEBUG("Plugin Op Cache: TensorDesc match");
    return true;
}

std::shared_ptr<AclNNTensor> CreateTensor(atb::Tensor atbTensor, int tensorIdx)
{
    std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
    aclnnTensor->needUpdateTensorDataPtr = true;
    aclnnTensor->atbTensor = atbTensor;
    aclnnTensor->tensorIdx = tensorIdx;
    aclnnTensor->strides = GetCopyTensorStride(atbTensor.desc.shape);
    CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensor);
    return aclnnTensor;
}

int ConvertTensorToSeqLengths(atb::Tensor &tensor, aclIntArray *&actualSeqLengths)
{
    static std::vector<int64_t> seqLenCache;
    size_t dataSize = tensor.dataSize / 8;  // 8: int64 size
    if (seqLenCache.size() < dataSize) {
        seqLenCache.resize(dataSize);
    }
    if (memcpy_s(seqLenCache.data(), dataSize * 8, tensor.hostData, dataSize * 8) != 0) { // 8: int64 size
        ATB_SPEED_LOG_ERROR(" memcpy_s failed");
        return atb::ERROR_INTERNAL_ERROR;
    }
    if (actualSeqLengths != nullptr) {
        aclDestroyIntArray(actualSeqLengths);
        actualSeqLengths = nullptr;
    }
    actualSeqLengths = aclCreateIntArray(static_cast<int64_t *>(seqLenCache.data()), dataSize);
    return atb::NO_ERROR;
}
} // namespace common
} // namespace atb_speed