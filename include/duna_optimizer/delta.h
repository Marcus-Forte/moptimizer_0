#pragma once

#include <Eigen/Dense>
namespace duna_optimizer {

template <class Scalar, int Dim>
inline bool isDeltaSmall(const Eigen::Matrix<Scalar, Dim, 1>& delta) {
  Scalar epsilon = delta.array().abs().maxCoeff();
  if (epsilon < sqrt(std::numeric_limits<Scalar>::epsilon())) return true;
  return false;
}

}  // namespace duna_optimizer