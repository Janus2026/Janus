#pragma once

#include <torch/torch.h>

#include <vector>

#include "core/framework/state_dict/state_dict.h"
#include "core/util/json_reader.h"
#include "core/util/threadpool.h"
#include "model_loader.h"

namespace janus {

bool load_quant_cfg(const JsonReader& reader, QuantArgs& quant_args);

class HFModelLoader : public ModelLoader {
 public:
  HFModelLoader(const std::string& model_weights_path);

  std::unique_ptr<Tokenizer> tokenizer() const override;

  std::vector<std::unique_ptr<StateDict>>& get_state_dicts() override;

  int64_t get_total_weight_size() const override;

  int64_t get_non_decoder_weight_size() const override;

  int64_t get_max_decoder_layer_weight_size() const override;

 private:
  bool load_args(const std::string& model_weights_path);
  bool load_rec_vocab(const std::string& model_weights_path);
  bool load_model_args(const std::string& model_weights_path);
  bool load_quant_args(const std::string& model_weights_path);
  bool load_tokenizer_args(const std::string& model_weights_path);
  bool load_image_preprocessor_args(const std::string& model_weights_path);
  bool load_video_preprocessor_args(const std::string& model_weights_path);
  std::string model_weights_path() const override {
    return model_weights_path_;
  }
  std::string model_weights_path_;

  // sorted model weights files
  std::vector<std::string> model_weights_files_;

  // models weights tensors
  std::vector<std::unique_ptr<StateDict>> state_dicts_;

  std::unique_ptr<ThreadPool> threadpool_;
};
}  // namespace janus
