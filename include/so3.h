#pragma once
#include <Eigen/Dense>

namespace so3
{
    // 6DOF
    template <typename Scalar>
    void param2Matrix(const Eigen::Matrix<Scalar, 6, 1> &x, Eigen::Matrix<Scalar,4,4>& transform_matrix_);


    // 3DOF
    template <typename Scalar>
    void param2Matrix(const Eigen::Matrix<Scalar, 3, 1> &x, Eigen::Matrix<Scalar,4,4>& transform_matrix_);

    template <typename Scalar>
    void matrix2Param(const Eigen::Matrix<Scalar, 4, 4> &transform_matrix_, Eigen::Matrix<Scalar, 6, 1> &x);

}