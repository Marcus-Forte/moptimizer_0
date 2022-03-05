#pragma once
#include <Eigen/Dense>

namespace so3
{
    template <typename Scalar>
    void param2Matrix(const Eigen::Matrix<float, 6, 1> &x, Eigen::Matrix<Scalar,4,4>& transform_matrix_);

}