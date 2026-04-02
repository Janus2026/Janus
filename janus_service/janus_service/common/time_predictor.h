#pragma once

#include <Eigen/Dense>
#include <unordered_map>
#include <glog/logging.h>

namespace janus_service {

// Predictor for predicting TTFT and TPOT
class TimePredictor final {
 public:
  TimePredictor(
      const std::unordered_map<std::string, std::vector<std::pair<int32_t, double>>>& ttft_profiling_data,
      const std::unordered_map<std::string, std::vector<std::tuple<int32_t, int32_t, double>>>&
          tpot_profiling_data);
  ~TimePredictor() = default;

  double predict_ttft(const std::string& model_id, int32_t length);
  double predict_tpot(const std::string& model_id, int32_t total_length, int32_t batch_size);

  // Legacy methods for backward compatibility (use default model)
  double predict_ttft(int32_t length);
  double predict_tpot(int32_t total_length, int32_t batch_size);

 private:
  std::unordered_map<std::string, Eigen::VectorXd> ttft_coefficients_;
  std::unordered_map<std::string, Eigen::VectorXd> tpot_coefficients_;
  
  // Default model for legacy compatibility
  std::string default_model_id_;
};

}  // namespace janus_service