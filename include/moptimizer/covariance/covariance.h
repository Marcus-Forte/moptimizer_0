#pragma once

#include <Eigen/Dense>
#include <memory>

#include "moptimizer/types.h"

namespace moptimizer::covariance {

template <class Scalar>
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
template <class Scalar>
using MatrixPtr = std::shared_ptr<Matrix<Scalar>>;

}  // namespace moptimizer::covariance
