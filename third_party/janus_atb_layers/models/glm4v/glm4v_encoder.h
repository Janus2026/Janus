#ifndef ATB_SPEED_MODELS_GLM4V_ENCODER_H
#define ATB_SPEED_MODELS_GLM4V_ENCODER_H

#include <vector>

#include "atb/atb_infer.h"
#include "atb_speed/base/hosttensor_binder.h"
#include "nlohmann/json.hpp"

namespace atb_speed {
namespace glm {

struct VisionEncoderLayerParam {
  bool isBF16 = false;
  bool supportLcoc = false;
  bool supportLora = false;
  bool loraEnableGMM = false;
  bool enableLogN = false;
  std::string backend = "hccl";
  int rank = 0;
  int worldSize = 1;
  int quantType = 0;
  int quantGroupSize = 64;
  int numAttentionHeadsPerRank = 0;
  int hiddenSizePerAttentionHead = 0;
  int numKeyValueHeadsPerRank = 0;
  float rmsNormEps = 0;

  std::vector<int> seqLen;
  std::vector<int> tokenOffset;
  std::vector<int> packQuantType = {};
  std::vector<int> linearQuantType = {};
  std::vector<int> linearTransposeType;
};

atb::Status Glm4v_EncoderLayer(const VisionEncoderLayerParam& param,
                         atb::Operation** operation);

}  // namespace glm
}  // namespace atb_speed
#endif