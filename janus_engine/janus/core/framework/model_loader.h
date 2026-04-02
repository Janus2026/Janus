#pragma once

#include <torch/torch.h>

#include <vector>

#include "core/framework/model/model_args.h"
#include "core/framework/quant_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/tokenizer/tokenizer.h"
#include "core/framework/tokenizer/tokenizer_args.h"

namespace janus {

class ModelLoader {
 public:
  enum ModelType : uint8_t { HF_MODEL_TYPE = 0, INVALID = 1 };

  virtual ~ModelLoader() = default;

  virtual const ModelArgs& model_args() const { return args_; }
  virtual const QuantArgs& quant_args() const { return quant_args_; }
  virtual const TokenizerArgs& tokenizer_args() const {
    return tokenizer_args_;
  }
  virtual std::unique_ptr<Tokenizer> tokenizer() const = 0;
  virtual std::vector<std::unique_ptr<StateDict>>& get_state_dicts() = 0;
  virtual std::string model_weights_path() const = 0;
  virtual int64_t get_total_weight_size() const { return 0; }

  // Returns the total byte size of non-decoder-layer weights
  // (embed_tokens, norm, lm_head). Used by rolling load to size XTensor
  // weight pages without including decoder layer weights.
  virtual int64_t get_non_decoder_weight_size() const { return 0; }

  // Returns the maximum byte size among all decoder layers.
  // Return -1 if unsupported or failed to compute.
  virtual int64_t get_max_decoder_layer_weight_size() const { return -1; }

 protected:
  // model args
  ModelArgs args_;
  // quantization args
  QuantArgs quant_args_;
  // tokenizer args
  TokenizerArgs tokenizer_args_;

 public:
  // create a model loader from the given path
  static std::unique_ptr<ModelLoader> create(
      const std::string& model_weights_path);
};

}  // namespace janus
