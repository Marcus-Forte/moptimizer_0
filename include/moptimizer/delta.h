#pragma once

#include <Eigen/Dense>
namespace moptimizer {

/// @brief Check if vector is _small enough_.
/// @tparam Scalar
/// @tparam Dim
/// @param delta vector to be checked.
/// @return
template <class Scalar, int Dim>
inline bool isDeltaSmall(const Eigen::Matrix<Scalar, Dim, 1>& delta) {
  Scalar epsilon = delta.array().abs().maxCoeff();
  if (epsilon < sqrt(std::numeric_limits<Scalar>::epsilon())) return true;
  return false;
}

}  // namespace moptimizer