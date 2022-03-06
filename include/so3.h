#pragma once
#include <Eigen/Dense>

namespace so3
{
    template <typename Scalar>
    void param2Matrix6DOF(const Eigen::Matrix<Scalar, 6, 1> &x, Eigen::Matrix<Scalar,4,4>& transform_matrix_);

    template <typename Scalar>
    void matrix2Param6DOF(const Eigen::Matrix<Scalar, 4, 4> &transform_matrix_, Eigen::Matrix<Scalar, 6, 1> &x);

}