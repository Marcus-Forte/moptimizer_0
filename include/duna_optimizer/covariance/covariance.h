#pragma once

#include <Eigen/Dense>
#include <memory>

#include "duna_optimizer/types.h"

namespace duna_optimizer::covariance {

template <class Scalar>
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
template <class Scalar>
using MatrixPtr = std::shared_ptr<Matrix<Scalar>>;

}  // namespace duna_optimizer::covariance
