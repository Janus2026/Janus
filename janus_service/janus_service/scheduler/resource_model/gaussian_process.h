#pragma once

#include <Eigen/Dense>
#include <utility>

#include "common/macros.h"

namespace janus_service {

// Gaussian Process regression with RBF (Squared Exponential) kernel and
// Automatic Relevance Determination (ARD) length scales.
//
// Kernel: k(x, x') = sigma_f^2 * exp(-0.5 * sum_d((x_d - x'_d)^2 / l_d^2))
//
// Training data and hyperparameters are provided at construction time.
// Hyperparameters are fitted offline by maximizing the log marginal likelihood
// (type-II MLE, see fit_gp_mle.py) and loaded from the GP data JSON.
// An optional constant prior mean (prior_mean) supports mean-centered fits:
// the GP is trained on (y - prior_mean) and prior_mean is added back at
// prediction time. prior_mean = 0 reproduces the legacy zero-mean behavior.
//
// The initial kernel matrix inverse (K + noise*I)^{-1} is formed once at
// construction via a Cholesky-based inverse. New observations can then be
// folded in at runtime with add_point(), which performs an O(n^2) incremental
// Sherman-Morrison (bordered / Schur-complement) rank-1 update of the explicit
// inverse instead of refactorizing the whole matrix. Note: add_point() mutates
// the model and is NOT thread-safe with respect to concurrent predict() calls;
// callers must serialize updates against predictions.
class GaussianProcess {
 public:
  // Construct from training data and hyperparameters.
  // X_train:        (N x D) matrix of training inputs
  // y_train:        (N) vector of training outputs
  // lengthscales:   (D) vector of ARD length scales
  // signal_variance: kernel signal variance (sigma_f^2)
  // noise_variance:  observation noise variance (sigma_n^2)
  // prior_mean:      constant prior mean subtracted from y_train (default 0)
  GaussianProcess(const Eigen::MatrixXd& X_train,
                  const Eigen::VectorXd& y_train,
                  const Eigen::VectorXd& lengthscales,
                  double signal_variance,
                  double noise_variance,
                  double prior_mean = 0.0);

  ~GaussianProcess() = default;

  // Predict mean at a single test point (D-dimensional).
  double predict_mean(const Eigen::VectorXd& x_test) const;

  // Predict mean and variance at a single test point.
  std::pair<double, double> predict(const Eigen::VectorXd& x_test) const;

  // Incrementally fold a new observation (x_new, y_new) into the model.
  // Updates the explicit inverse K^{-1} and alpha in O(n^2) via a bordered
  // Sherman-Morrison (Schur-complement) rank-1 update, plus refreshes the
  // input/output clamping bounds. NOT thread-safe against predict().
  void add_point(const Eigen::VectorXd& x_new, double y_new);

  int input_dim() const { return input_dim_; }
  int num_train() const { return n_train_; }

 private:
  DISALLOW_COPY_AND_ASSIGN(GaussianProcess);

  // RBF-ARD kernel value between two points.
  double rbf_kernel(const Eigen::VectorXd& x1,
                    const Eigen::VectorXd& x2) const;

  // Kernel vector between a test point and all training points.
  Eigen::VectorXd kernel_vector(const Eigen::VectorXd& x_test) const;

  int input_dim_;
  int n_train_;

  // Hyperparameters
  Eigen::VectorXd lengthscales_;
  double signal_variance_;
  double noise_variance_;
  double prior_mean_;

  // Training data
  Eigen::MatrixXd X_train_;  // (N x D)
  Eigen::VectorXd y_train_;  // (N,)

  // Training data bounds for input clamping (prevents extrapolation collapse)
  Eigen::VectorXd x_min_;  // (D,) per-dimension minimum of X_train
  Eigen::VectorXd x_max_;  // (D,) per-dimension maximum of X_train

  // Output lower bound: predictions below y_min_ are clamped up to y_min_
  double y_min_;

  // Precomputed for fast prediction
  Eigen::MatrixXd K_inv_;  // Explicit (K_train + noise*I)^{-1}, maintained
                           // incrementally via Sherman-Morrison in add_point()
  Eigen::VectorXd alpha_;  // K_train^{-1} * (y_train - prior_mean)
};

}  // namespace janus_service
