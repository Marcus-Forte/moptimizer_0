#pragma once
#include <Eigen/Dense>

#define SKEW_SYMMETRIC_FROM(v) 0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0

namespace so3
{
    template <typename Scalar>
    void convert6DOFParameterToMatrix(const Scalar *x, Eigen::Matrix<Scalar, 4, 4> &transform_matrix_);

    template <typename Scalar>
    void convert3DOFParameterToMatrix(const Scalar *x, Eigen::Matrix<Scalar, 4, 4> &transform_matrix_);

    template <typename Scalar>
    void convert3DOFParameterToMatrix3(const Scalar *x, Eigen::Matrix<Scalar, 3, 3> &transform_matrix_);

    template <typename Scalar>
    void convertMatrixTo6DOFParameter(const Eigen::Matrix<Scalar, 4, 4> &transform_matrix_, Eigen::Matrix<Scalar, 6, 1> &x);

    template <typename Scalar>
    void convertMatrixTo3DOFParameter(const Eigen::Matrix<Scalar, 4, 4> &transform_matrix_, Eigen::Matrix<Scalar, 3, 1> &x);

    /* Performs R = exp(delta). This means R ⊞ delta*/
    template <typename Scalar>
    void Exp(const Eigen::Ref<const Eigen::Matrix<Scalar, 3, 1>> &delta, Eigen::Ref<Eigen::Matrix<Scalar, 3, 3>> R);

    template <typename Scalar>
    Eigen::Matrix<Scalar, 3, 3> Exp(const Eigen::Matrix<Scalar, 3, 1> &ang);

    template <typename Scalar>
    Eigen::Matrix<Scalar, 3, 3> Exp(const Eigen::Matrix<Scalar, 3, 1> &ang_vel, const Scalar &dt);

    /* Performs delta = LOG(R). This means R ⊟ delta*/
    template <typename Scalar>
    void Log(const Eigen::Ref<Eigen::Matrix<Scalar, 3, 3>> &R, Eigen::Matrix<Scalar, 3, 1> &delta);

    template <typename Scalar>
    void inverseRightJacobian(const Eigen::Matrix<Scalar, 3, 1> &r, Eigen::Ref<Eigen::Matrix<Scalar, 3, 3>> inv_jacobian);
}