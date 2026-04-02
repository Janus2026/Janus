// /*
//  * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
//  *
//  * Licensed under the Apache License, Version 2.0 (the "License");
//  * you may not use this file except in compliance with the License.
//  * You may obtain a copy of the License at
//  *
//  * http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing, software
//  * distributed under the License is distributed on an "AS IS" BASIS,
//  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  * See the License for the specific language governing permissions and
//  * limitations under the License.
//  */
#ifndef ATB_SPEED_PLUGIN_ACLNN_MULTI_LATENT_ATTENTION_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_MULTI_LATENT_ATTENTION_OPERATION_H
#include "operations/aclnn/core/acl_nn_operation.h"
#include "operations/aclnn/core/acl_nn_operation_cache.h"
#include "atb_speed/utils/operation_util.h"
#include <cstdint>

namespace atb_speed {
namespace common {
// enum MaskType {
//     MASK_TYPE_NONE = 0, 
//     MASK_TYPE_NORM = 1,
//     MASK_TYPE_ALIBI = 2,
//     MASK_TYPE_LOOK_AHEAD = 3,
//     MASK_TYPE_MASK_FREE = 4
// };

// enum CalcType : int {
//     CALC_TYPE_UNDEFINED = 0, // default value
//     CALC_TYPE_SPEC,          // seqlen > 1 (a.k.a speculative decode)
//     CALC_TYPE_RING,          // ringAttention
// };

// ! \brief MultiLatentAttentiontranslated
// !
// ! Copy from atb::infer::MultiLatentAttentionParam
struct AclNNMultiLatentAttentionParam {
    //!
    //! \brief querytranslated
    //!
    int32_t headNum = 0;
    //!
    //! \brief translatedtortranslated, translatedQ*K^Ttranslated
    //!
    float qkScale = 1.0;
    //!
    //! \brief kvtranslated
    //!
    int32_t kvHeadNum = 0;
    //!
    //! \enum MaskType
    //!
    //! \brief The type values of MaskType.
    //!
    enum MaskType : int {
        UNDEFINED = 0,       //!< translated,translated0translatedmask
        MASK_TYPE_SPEC,      //!< qseqlen > 1translatedmask
        MASK_TYPE_MASK_FREE, //!< mask free
    };
    //!
    //! \brief masktranslated
    //!
    MaskType maskType = UNDEFINED;
    //!
    //! \enum CalcType
    //!
    //! \brief The type values of CalcType.
    //!
    enum CalcType : int {
        CALC_TYPE_UNDEFINED = 0, // translated
        CALC_TYPE_SPEC,          // translated1translatedqseqlen
        CALC_TYPE_RING,          // ringAttention
    };
    //!
    //! \brief CalcTypetranslated
    //!
    CalcType calcType = CALC_TYPE_UNDEFINED;
    //!
    //! \enum CacheMode
    //!
    //! \brief translatedcachetranslated.
    //!
    enum CacheMode : uint8_t {
        KVCACHE = 0,  // translatedcache
        KROPE_CTKV,   // translatedcache,translated
        INT8_NZCACHE, // translatedcache
        NZCACHE,      // translatedNZcache
    };
    //!
    //! \brief translatedcachetranslated.
    //!
    CacheMode cacheMode = KVCACHE;
};
// struct AclNNMultiLatentAttentionParam{
//     int type;
//     int headSize;
//     float tor;
//     int kvHead;
//     int maskType = 0;
//     int calcType = 0;
//     std::vector<int32_t> qSeqLen;
//     std::vector<int32_t> kvSeqLen;
//     int isRing = 0;
// }

class MultiLatentAttentionOperation : public AclNNOperation {
public:
    explicit MultiLatentAttentionOperation(const std::string &name, AclNNMultiLatentAttentionParam param);
    ~MultiLatentAttentionOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;
private:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;
    atb::Status CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack) override;
    int BuildFromTensor(const atb::VariantPack &variantPack, size_t contextLensTensorId, size_t qSeqlenTensorId, bool needQLens);
    AclNNMultiLatentAttentionParam param_;
    std::vector<int32_t> qSeqLen;
    std::vector<int32_t> kvSeqLen;
    static const uint32_t IN_TENSOR_NUM = 6;
    static const uint32_t OUT_TENSOR_NUM_1 = 1;
    static const uint32_t OUT_TENSOR_NUM_2 = 2;
    static const uint32_t CONTEXTLENS_INDEX = 5;
    static const uint32_t QSEQLEN_INDEX = 6;
};
}  // namespace common
}  // namespace atb_speed
#endif  // ATB_SPEED_PLUGIN_ACLNN_MULTI_LATENT_ATTENTION_OPERATION_H