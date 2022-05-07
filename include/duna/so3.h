#pragma once
#include <Eigen/Dense>

namespace so3
{
    template <typename ScalarIn, typename Scalar>
    void convert6DOFParameterToMatrix(const ScalarIn* x, Eigen::Matrix<Scalar,4,4>& transform_matrix_);

    template <typename Scalar>
    void convert3DOFParameterToMatrix(const Scalar* x, Eigen::Matrix<Scalar,4,4>& transform_matrix_);

    template <typename Scalar>
    void convertMatrixTo6DOFParameter(const Eigen::Matrix<Scalar, 4, 4> &transform_matrix_, Eigen::Matrix<Scalar, 6, 1> &x);
}