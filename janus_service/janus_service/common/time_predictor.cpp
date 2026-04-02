#include "time_predictor.h"

static constexpr int32_t kDegree = 2;

namespace janus_service {

TimePredictor::TimePredictor(
    const std::unordered_map<std::string, std::vector<std::pair<int32_t, double>>>& ttft_profiling_data,
    const std::unordered_map<std::string, std::vector<std::tuple<int32_t, int32_t, double>>>&
        tpot_profiling_data) {
  
  // Process TTFT profiling data for each model
  for (const auto& [model_id, data] : ttft_profiling_data) {
    if (!data.empty()) {
      // construct Vandermonde matrix
      int32_t m = data.size();
      int32_t n = kDegree + 1;
      Eigen::MatrixXd matrix(m, n);
      for (int32_t i = 0; i < m; ++i) {
        for (int32_t j = 0; j < n; ++j) {
          matrix(i, j) = std::pow(data[i].first, j);
        }
      }

      // construct target vector
      Eigen::VectorXd target(m);
      for (int32_t i = 0; i < m; ++i) {
        target(i) = data[i].second;
      }

      // get coefficients
      ttft_coefficients_[model_id] = matrix.colPivHouseholderQr().solve(target);
    } else {
      ttft_coefficients_[model_id] = Eigen::VectorXd::Zero(1);
    }
  }

  // Process TPOT profiling data for each model
  for (const auto& [model_id, data] : tpot_profiling_data) {
    if (!data.empty()) {
      int32_t m = data.size();
      int32_t n = kDegree + 1;
      Eigen::MatrixXd matrix(m, n);
      for (int32_t i = 0; i < m; ++i) {
        int32_t avg_length = std::get<0>(data[i]);
        int32_t batch_size = std::get<1>(data[i]);

        matrix(i, 0) = 1.0;  // the index 0 is always for constant
        matrix(i, 1) = batch_size;
        matrix(i, 2) = batch_size * (avg_length - 1);
      }

      // construct target vector
      Eigen::VectorXd target(m);
      for (int32_t i = 0; i < m; ++i) {
        target(i) = std::get<2>(data[i]);
      }

      // get coefficients
      tpot_coefficients_[model_id] = matrix.colPivHouseholderQr().solve(target);
    } else {
      tpot_coefficients_[model_id] = Eigen::VectorXd::Zero(3);
    }
  }

  // Set default model for backward compatibility
  if (!ttft_profiling_data.empty()) {
    default_model_id_ = ttft_profiling_data.begin()->first;
  } else if (!tpot_profiling_data.empty()) {
    default_model_id_ = tpot_profiling_data.begin()->first;
  }
}

double TimePredictor::predict_ttft(const std::string& model_id, int32_t length) {
  auto it = ttft_coefficients_.find(model_id);
  if (it == ttft_coefficients_.end()) {
    if (!default_model_id_.empty() && ttft_coefficients_.find(default_model_id_) != ttft_coefficients_.end()) {
      it = ttft_coefficients_.find(default_model_id_);
    } else if (!ttft_coefficients_.empty()) {
      it = ttft_coefficients_.begin();
    } else {
      // Default: ttft_ms = 0.0678 * length + 19.63 (fitted from profiling)
      return 0.0678 * length + 19.63;
    }
  }

  const Eigen::VectorXd& coefficients = it->second;
  double result = 0.0;
  double power = 1.0;
  for (int32_t i = 0; i < coefficients.size(); ++i) {
    result += coefficients(i) * power;
    power *= length;
  }

  return result;
}

double TimePredictor::predict_tpot(const std::string& model_id, int32_t total_length, int32_t batch_size) {
  auto it = tpot_coefficients_.find(model_id);
  if (it == tpot_coefficients_.end()) {
    LOG(WARNING) << "Model " << model_id << " not found in TPOT coefficients, using default";
    if (!default_model_id_.empty() && tpot_coefficients_.find(default_model_id_) != tpot_coefficients_.end()) {
      it = tpot_coefficients_.find(default_model_id_);
    } else if (!tpot_coefficients_.empty()) {
      it = tpot_coefficients_.begin();
    } else {
      return 0.0;
    }
  }
  
  const Eigen::VectorXd& coefficients = it->second;
  double result = coefficients(0) + coefficients(1) * batch_size +
                  coefficients(2) * total_length;
  return result;
}

// Legacy methods for backward compatibility
double TimePredictor::predict_ttft(int32_t length) {
  if (!default_model_id_.empty()) {
    return predict_ttft(default_model_id_, length);
  } else if (!ttft_coefficients_.empty()) {
    return predict_ttft(ttft_coefficients_.begin()->first, length);
  }
  return 0.0;
}

double TimePredictor::predict_tpot(int32_t total_length, int32_t batch_size) {
  if (!default_model_id_.empty()) {
    return predict_tpot(default_model_id_, total_length, batch_size);
  } else if (!tpot_coefficients_.empty()) {
    return predict_tpot(tpot_coefficients_.begin()->first, total_length, batch_size);
  }
  return 0.0;
}

}  // namespace janus_service