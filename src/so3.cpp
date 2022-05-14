#include "duna/so3.h"
#include "duna_exports.h"
#include <iostream>
namespace so3
{
    // TODO check for errors in conversion. High angles are problematic
    template <typename Scalar>
    inline void convert6DOFParameterToMatrix(const Scalar *x, Eigen::Matrix<Scalar, 4, 4> &transform_matrix_)
    {
        transform_matrix_.setZero();
        transform_matrix_(0, 3) = x[0];
        transform_matrix_(1, 3) = x[1];
        transform_matrix_(2, 3) = x[2];
        transform_matrix_(3, 3) = 1;

        // Compute w from the unit quaternion
        // Eigen::Quaternion<Scalar> q(0, x[3], x[4], x[5]);
        // Scalar &&q_dot_q = q.dot(q);
        // q.w() = static_cast<Scalar>(std::sqrt(1 - q_dot_q));
        // q.normalize();
        // transform_matrix_.topLeftCorner(3, 3) = q.toRotationMatrix();

        // EXP
        Eigen::Matrix<Scalar, 3, 1> delta(x[3], x[4], x[5]);
        delta = 2 * delta; // TODO why ?
        Eigen::Matrix<Scalar, 3, 3> rot;
        Exp<Scalar>(delta, rot);
        transform_matrix_.topLeftCorner(3, 3) = rot;
    }

    template <typename Scalar>
    inline void convert3DOFParameterToMatrix(const Scalar *x, Eigen::Matrix<Scalar, 4, 4> &transform_matrix_)
    {
        transform_matrix_.setZero();
        transform_matrix_(3, 3) = 1;

        // Compute w from the unit quaternion
        Eigen::Quaternion<Scalar> q(0, x[0], x[1], x[2]);
        q.w() = static_cast<Scalar>(std::sqrt(1 - q.dot(q)));
        q.normalize();
        transform_matrix_.topLeftCorner(3, 3) = q.toRotationMatrix();
    }

    template <typename Scalar>
    inline void convertMatrixTo6DOFParameter(const Eigen::Matrix<Scalar, 4, 4> &transform_matrix_, Eigen::Matrix<Scalar, 6, 1> &x)
    {
        x[0] = transform_matrix_(0, 3);
        x[1] = transform_matrix_(1, 3);
        x[2] = transform_matrix_(2, 3);

        Eigen::Matrix<Scalar, 3, 1> ea;

        Eigen::Matrix<Scalar, 3, 3> rot = transform_matrix_.topLeftCorner(3, 3);
        ea = rot.eulerAngles(2, 1, 0);
        x[3] = ea[0];
        x[4] = ea[1]; //;ea[1];
        x[5] = ea[2]; // ea[2];
    }

    template <typename Scalar>
    void Exp(const Eigen::Ref<const Eigen::Matrix<Scalar, 3, 1>> &delta, Eigen::Ref<Eigen::Matrix<Scalar, 3, 3>> R)
    {
        Eigen::Vector3d delta_conv(delta[0],delta[1],delta[2]);
        double theta_sq = delta_conv.dot(delta_conv);

        double theta;
        double imag_factor;
        double real_factor;
        if (theta_sq < 1e-10)
        {
            theta = 0;
            double theta_quad = theta_sq * theta_sq;
            imag_factor = 0.5 - 1.0 / 48.0 * theta_sq + 1.0 / 3840.0 * theta_quad;
            real_factor = 1.0 - 1.0 / 8.0 * theta_sq + 1.0 / 384.0 * theta_quad;
        }
        else
        {
            theta = std::sqrt(theta_sq);
            double half_theta = 0.5 * theta;
            imag_factor = std::sin(half_theta) / theta;
            real_factor = std::cos(half_theta);
        }
        
        Eigen::Quaterniond q(real_factor, imag_factor * delta_conv[0], imag_factor * delta_conv[1], imag_factor * delta_conv[2]);
        R = q.toRotationMatrix().template cast<Scalar>();

        // Eigen::Vector3d delta_conv(delta[0], delta[1], delta[2]);
        // double delta_norm = delta_conv.norm();
        // Eigen::Matrix3d Eye = Eigen::Matrix3d::Identity();

        // if (delta_norm > 1e-10)
        // {
        //     Eigen::Vector3d r_axis = delta_conv / delta_norm;
        //     Eigen::Matrix3d K;

        //     K << SKEW_SYM_MATRX(r_axis);

        //     R = (Eye + std::sin(delta_norm) * K + (1.0 - std::cos(delta_norm)) * K * K).template cast<Scalar>();
        // }
        // else
        // {
        //     R.setIdentity();
        // }
    }

    template <typename Scalar>
    void Log(const Eigen::Ref<Eigen::Matrix<Scalar, 3, 3>> &R, Eigen::Matrix<Scalar, 3, 1> &delta)
    {
        Scalar theta = (R.trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (R.trace() - 1));
        Eigen::Matrix<Scalar, 3, 1> K(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
        if (std::abs(theta) < 0.001)
        {
            delta = (0.5 * K);
        }
        else
        {
            delta = (0.5 * theta / std::sin(theta) * K);
        }
    }

    template void DUNA_OPTIMIZER_EXPORT convert6DOFParameterToMatrix<double>(const double *x, Eigen::Matrix<double, 4, 4> &transform_matrix_);
    template void DUNA_OPTIMIZER_EXPORT convert6DOFParameterToMatrix<float>(const float *x, Eigen::Matrix<float, 4, 4> &transform_matrix_);

    template void DUNA_OPTIMIZER_EXPORT Exp<double>(const Eigen::Ref<const Eigen::Matrix<double, 3, 1>> &delta, Eigen::Ref<Eigen::Matrix<double, 3, 3>> R);
    template void DUNA_OPTIMIZER_EXPORT Exp<float>(const Eigen::Ref<const Eigen::Matrix<float, 3, 1>> &delta, Eigen::Ref<Eigen::Matrix<float, 3, 3>> R);

    template void DUNA_OPTIMIZER_EXPORT Log<double>(const Eigen::Ref<Eigen::Matrix<double, 3, 3>> &R, Eigen::Matrix<double, 3, 1> &delta);
    template void DUNA_OPTIMIZER_EXPORT Log<float>(const Eigen::Ref<Eigen::Matrix<float, 3, 3>> &R, Eigen::Matrix<float, 3, 1> &delta);
}
