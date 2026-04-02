#include "gaussian_process.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>

namespace janus_service {

GaussianProcess::GaussianProcess(const Eigen::MatrixXd& X_train,
                                 const Eigen::VectorXd& y_train,
                                 const Eigen::VectorXd& lengthscales,
                                 double signal_variance,
                                 double noise_variance,
                                 double prior_mean)
    : input_dim_(X_train.cols()),
      n_train_(X_train.rows()),
      lengthscales_(lengthscales),
      signal_variance_(signal_variance),
      noise_variance_(noise_variance),
      prior_mean_(prior_mean),
      X_train_(X_train),
      y_train_(y_train),
      x_min_(X_train.colwise().minCoeff()),
      x_max_(X_train.colwise().maxCoeff()),
      y_min_(y_train.minCoeff()) {
  CHECK_EQ(X_train.rows(), y_train.size())
      << "X_train rows must match y_train size";
  CHECK_EQ(lengthscales.size(), input_dim_)
      << "Lengthscales dimension must match input dimension";
  CHECK_GT(signal_variance, 0.0) << "Signal variance must be positive";
  CHECK_GE(noise_variance, 0.0) << "Noise variance must be non-negative";

  // Build kernel matrix K_train (N x N)
  Eigen::MatrixXd K(n_train_, n_train_);
  for (int i = 0; i < n_train_; ++i) {
    for (int j = i; j < n_train_; ++j) {
      double k_val = rbf_kernel(X_train_.row(i), X_train_.row(j));
      K(i, j) = k_val;
      K(j, i) = k_val;
    }
    // Add noise variance to diagonal
    K(i, i) += noise_variance_;
  }

  // Cholesky decomposition: K = L * L^T, used only to form a numerically
  // stable initial inverse. Subsequent add_point() calls update the inverse
  // incrementally via Sherman-Morrison rather than refactorizing.
  Eigen::LLT<Eigen::MatrixXd> llt(K);
  if (llt.info() != Eigen::Success) {
    // Add jitter for numerical stability and retry
    LOG(WARNING) << "GP Cholesky failed, adding jitter (1e-6) to diagonal";
    K.diagonal().array() += 1e-6;
    llt.compute(K);
    CHECK(llt.info() == Eigen::Success)
        << "GP Cholesky decomposition failed even with jitter";
  }

  // Explicit inverse K^{-1} = (K_train + noise*I)^{-1}, maintained
  // incrementally by add_point().
  K_inv_ = llt.solve(Eigen::MatrixXd::Identity(n_train_, n_train_));

  // Precompute alpha = K^{-1} * (y_train - prior_mean)
  alpha_ = K_inv_ * (y_train_.array() - prior_mean_).matrix();

  LOG(INFO) << "GaussianProcess initialized: " << n_train_ << " training points"
            << ", input_dim=" << input_dim_
            << ", signal_var=" << signal_variance_
            << ", noise_var=" << noise_variance_
            << ", prior_mean=" << prior_mean_
            << ", x_min=[" << x_min_.transpose() << "]"
            << ", x_max=[" << x_max_.transpose() << "]"
            << ", y_min=" << y_min_;
}

double GaussianProcess::rbf_kernel(const Eigen::VectorXd& x1,
                                   const Eigen::VectorXd& x2) const {
  // k(x1, x2) = sigma_f^2 * exp(-0.5 * sum_d((x1_d - x2_d)^2 / l_d^2))
  Eigen::VectorXd diff = (x1 - x2).array() / lengthscales_.array();
  return signal_variance_ * std::exp(-0.5 * diff.squaredNorm());
}

Eigen::VectorXd GaussianProcess::kernel_vector(
    const Eigen::VectorXd& x_test) const {
  Eigen::VectorXd k_star(n_train_);
  for (int i = 0; i < n_train_; ++i) {
    k_star(i) = rbf_kernel(x_test, X_train_.row(i));
  }
  return k_star;
}

double GaussianProcess::predict_mean(const Eigen::VectorXd& x_test) const {
  // Clamp input to training data range to prevent extrapolation collapse
  Eigen::VectorXd x_clamped = x_test.cwiseMax(x_min_).cwiseMin(x_max_);
  Eigen::VectorXd k_star = kernel_vector(x_clamped);
  double mean = prior_mean_ + k_star.dot(alpha_);
  // Clamp output: predictions below training y_min are unreliable
  return std::max(mean, y_min_);
}

std::pair<double, double> GaussianProcess::predict(
    const Eigen::VectorXd& x_test) const {
  // Clamp input to training data range to prevent extrapolation collapse
  Eigen::VectorXd x_clamped = x_test.cwiseMax(x_min_).cwiseMin(x_max_);
  Eigen::VectorXd k_star = kernel_vector(x_clamped);

  // Mean: prior_mean + k*^T * alpha
  double mean = prior_mean_ + k_star.dot(alpha_);
  // Clamp output: predictions below training y_min are unreliable
  mean = std::max(mean, y_min_);

  // Variance: k(x*, x*) - k*^T * K^{-1} * k*
  Eigen::VectorXd w = K_inv_ * k_star;
  double k_ss = signal_variance_;  // k(x*, x*) = sigma_f^2 (self-kernel)
  double variance = k_ss - k_star.dot(w);

  // Clamp variance to non-negative (numerical stability)
  variance = std::max(variance, 0.0);

  return {mean, variance};
}

void GaussianProcess::add_point(const Eigen::VectorXd& x_new, double y_new) {
  CHECK_EQ(x_new.size(), input_dim_)
      << "add_point input dimension must match GP input dimension";

  const int n = n_train_;

  // b = k(x_new, X_train_)  (n,) ; c = k(x_new, x_new) + noise (scalar)
  Eigen::VectorXd b(n);
  for (int i = 0; i < n; ++i) {
    b(i) = rbf_kernel(x_new, X_train_.row(i));
  }
  const double c = signal_variance_ + noise_variance_;

  // Bordered (Schur-complement) rank-1 update of K^{-1}:
  //   K_{n+1} = [[K_n,  b ],
  //              [b^T,  c ]]
  //   v = K_n^{-1} b
  //   s = c - b^T v                         (Schur complement, scalar)
  //   K_{n+1}^{-1} = [[K_n^{-1} + v v^T / s,  -v / s],
  //                   [        -v^T / s,        1 / s]]
  Eigen::VectorXd v = K_inv_ * b;  // (n,)  O(n^2)
  double s = c - b.dot(v);
  if (s <= 1e-12) {
    // Near-duplicate input makes the kernel matrix singular; add jitter so the
    // Schur complement stays positive and the inverse remains well-defined.
    LOG(WARNING) << "GP add_point Schur complement non-positive (" << s
                 << "), clamping with jitter";
    s = std::max(s, 0.0) + 1e-6;
  }

  Eigen::MatrixXd K_inv_new(n + 1, n + 1);
  if (n > 0) {
    K_inv_new.topLeftCorner(n, n) = K_inv_ + (v * v.transpose()) / s;
    K_inv_new.topRightCorner(n, 1) = -v / s;
    K_inv_new.bottomLeftCorner(1, n) = (-v / s).transpose();
  }
  K_inv_new(n, n) = 1.0 / s;
  K_inv_ = std::move(K_inv_new);

  // Append (x_new, y_new) to the stored training data.
  X_train_.conservativeResize(n + 1, input_dim_);
  X_train_.row(n) = x_new.transpose();
  y_train_.conservativeResize(n + 1);
  y_train_(n) = y_new;

  // Refresh input/output clamping bounds.
  if (n == 0) {
    x_min_ = x_new;
    x_max_ = x_new;
    y_min_ = y_new;
  } else {
    x_min_ = x_min_.cwiseMin(x_new);
    x_max_ = x_max_.cwiseMax(x_new);
    y_min_ = std::min(y_min_, y_new);
  }

  n_train_ = n + 1;

  // Recompute alpha = K^{-1} * (y_train - prior_mean)  O(n^2)
  alpha_ = K_inv_ * (y_train_.array() - prior_mean_).matrix();
}

}  // namespace janus_service
