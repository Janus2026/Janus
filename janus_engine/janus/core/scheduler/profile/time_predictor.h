#pragma once

#include <Eigen/Dense>

namespace janus {

// Predictor for predicting time based on input length
class TimePredictor final {
 public:
  explicit TimePredictor(bool if_profile_prefix, bool is_prefill);

  void fit_for_prefill(
      const std::vector<std::pair<int32_t, double>>& time_profiling_data);

  void fit_for_prefill(const std::vector<std::tuple<int32_t, int32_t, double>>&
                           time_profiling_data);

  void fit_for_decode(const std::vector<std::tuple<int32_t, int32_t, double>>&
                          time_profiling_data);

  ~TimePredictor() = default;

  double predict_time(int32_t length,
                      int32_t prefix_length = 0,
                      bool if_need_add_constant_term = true);

  double get_constant_overhead();

  bool is_trained() { return trained_; }

  void check_coefficients_non_neg(int32_t num);

  int32_t get_quadratic_root(int32_t prefix_length, double budget);

  std::vector<double> get_coefficients();

 private:
  Eigen::VectorXd coefficients_;
  bool if_profile_prefix_ = false;
  bool is_prefill_ = false;
  bool trained_ = false;
};

}  // namespace janus
