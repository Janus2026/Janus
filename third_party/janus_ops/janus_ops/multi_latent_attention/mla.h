#ifndef ATBOPS_PARAMS_MLA_H
#define ATBOPS_PARAMS_MLA_H

#include <cstdint>
#include <string>
#include <sstream>
#include <vector>
#include "compare.h"

namespace AtbOps {
namespace OpParam {
struct MLA {
    enum Type {
        SPLIT_CACHE = 0,
    };
    Type type;
    int32_t headSize = 0;
    float tor = 0;
    int32_t kvHead = 0;

    enum MaskType {
        MASK_TYPE_NONE = 0,
        MASK_TYPE_NORM = 1,
        MASK_TYPE_ALIBI = 2,
        MASK_TYPE_LOOK_AHEAD = 3,
        MASK_TYPE_MASK_FREE = 4
    };

    MaskType maskType = MASK_TYPE_NONE;

    std::vector<int32_t> qSeqLen;
    std::vector<int32_t> kvSeqLen;

    int32_t isRing = 0;

    bool operator==(const MLA &other) const
    {
        return this->headSize == other.headSize &&
               this->qSeqLen == other.qSeqLen && this->kvSeqLen == other.kvSeqLen && this->type == other.type &&
               Utils::Compare<float>::IsEqual(this->tor, other.tor) && this->kvHead == other.kvHead &&
               this->isRing == other.isRing;
    }
};
} // namespace OpParam
} // namespace AtbOps
#endif // ATBOPS_PARAMS_MLA_H