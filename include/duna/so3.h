#pragma once
#include <Eigen/Dense>

#define SKEW_SYM_MATRX(v) 0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0

namespace so3
{
    template <typename Scalar>
    void convert6DOFParameterToMatrix(const Scalar *x, Eigen::Matrix<Scalar, 4, 4> &transform_matrix_);

    template <typename Scalar>
    void convert3DOFParameterToMatrix(const Scalar *x, Eigen::Matrix<Scalar, 4, 4> &transform_matrix_);

    template <typename Scalar>
    void convertMatrixTo6DOFParameter(const Eigen::Matrix<Scalar, 4, 4> &transform_matrix_, Eigen::Matrix<Scalar, 6, 1> &x);

    /* Performs R = exp(delta). This means R ⊞ delta*/
    template <typename Scalar>
    void Exp(const Eigen::Ref<const Eigen::Matrix<Scalar, 3, 1>> &delta, Eigen::Ref<Eigen::Matrix<Scalar, 3, 3>> R);

    /* Performs delta = LOG(R). This means R ⊟ delta*/
    template <typename Scalar>
    void Log(const Eigen::Ref<Eigen::Matrix<Scalar, 3, 3>>& R, Eigen::Matrix<Scalar, 3, 1> &delta );
}