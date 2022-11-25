#pragma once

#include <duna/cost_function.h>
#include <duna/so3.h>

namespace duna
{
    /* This is a special type of cost that computes the jacobian of the MAP problem with rotation states.
     See Eq. (23) from https://arxiv.org/pdf/2102.12400.pdf

    */
    template <typename Scalar>
    class CostFunctionRotationError : public CostFunctionBase<Scalar>
    {
    public:
        using RotationVector = Eigen::Matrix<Scalar, 3, 1>;
        using RotationMatrix = Eigen::Matrix<Scalar, 3, 3>;

        using StateVector = Eigen::Matrix<Scalar, 6, 1>;
        using StateMatrix = Eigen::Matrix<Scalar, 6, 6>;

    public:
        // Construct the rotation matrix with the initial states

        CostFunctionRotationError(const StateMatrix &covariance)
        {
            covariance_inverse_ = covariance.inverse();
        }
        CostFunctionRotationError()
        {
            covariance_inverse_.setIdentity();
        };

        // Initialize states and rotations at the beggining of the optimization. See Eq (15) from FASTLIO
        void setup(const Scalar *x) override
        {
            R_k_.setIdentity();
            RotationVector x_(x[0], x[1], x[2]);
            so3::convert3DOFParameterToMatrix3<Scalar>(x_.data(), R_k_);
            Eigen::Map<const StateVector> x_map(x);
            x_k_ = x_map;
        }

        Scalar computeCost(const Scalar *x) override
        {
            Eigen::Map<const StateVector> x_map(x);

            // Rotation matrix of ite
            RotationMatrix R_k_k;
            so3::convert3DOFParameterToMatrix3<Scalar>(x, R_k_k);

            RotationMatrix &&rot_diff = R_k_.transpose() * R_k_k;
            RotationVector diff_r;
            so3::Log<Scalar>(rot_diff, diff_r);

            StateVector x_diff;
            x_diff[0] = diff_r[0];
            x_diff[1] = diff_r[1];
            x_diff[2] = diff_r[2];
            x_diff[3] = x_map[3] - x_k_[3];
            x_diff[4] = x_map[4] - x_k_[4];
            x_diff[5] = x_map[5] - x_k_[5];

            Scalar sum = 2 * x_diff.transpose() * covariance_inverse_ * x_diff;

            return sum;
        }
        Scalar linearize(const Scalar *x, Scalar *hessian, Scalar *b) override
        {
            Eigen::Map<const StateVector> x_map(x);
            Eigen::Map<StateMatrix> hessian_map(hessian);
            Eigen::Map<StateVector> b_map(b);

            hessian_map.setZero();
            b_map.setZero();

            Eigen::Matrix<Scalar, 6, 6> jacobian;
            jacobian.setIdentity();

            // // Rotation matrix of ite
            RotationMatrix R_k_k;
            so3::convert3DOFParameterToMatrix3<Scalar>(x, R_k_k);

            // Calculate difference R_k_k - R_k in SO3 space.
            RotationMatrix &&diff = R_k_.transpose() * R_k_k;

            RotationVector diff_r;
            so3::Log<Scalar>(diff, diff_r);

            Eigen::Matrix<Scalar, 3, 3> right_inv_jacobian;
            so3::inverseRightJacobian<Scalar>(diff_r, right_inv_jacobian);

            // Note, this is already symmetric
            jacobian.template block<3, 3>(0, 0) = right_inv_jacobian;

            hessian_map = jacobian.transpose() * covariance_inverse_ * jacobian;
            // hessian_map.template selfadjointView<Eigen::Lower>().rankUpdate(jacobian.transpose()); // H = J^T * J
            // hessian_map.template triangularView<Eigen::Upper>() = hessian_map.transpose();

            StateVector x_diff;
            x_diff[0] = diff_r[0];
            x_diff[1] = diff_r[1];
            x_diff[2] = diff_r[2];
            x_diff[3] = x_map[3] - x_k_[3];
            x_diff[4] = x_map[4] - x_k_[4];
            x_diff[5] = x_map[5] - x_k_[5];

            b_map.noalias() = jacobian.transpose() * covariance_inverse_ * x_diff;

            return 2 * x_diff.transpose() * covariance_inverse_ * x_diff;
        }

    private:
        RotationMatrix R_k_;
        StateVector x_k_;
        StateMatrix covariance_inverse_;
    };
}