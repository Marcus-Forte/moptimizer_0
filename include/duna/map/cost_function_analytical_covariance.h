#pragma once

#include <duna/cost_function.h>
#include <duna/logger.h>

// TODO fix duplicates

namespace duna
{
    // NOTE. We are using Model as a template to be able to call its copy constructors and enable numercial diff.
    template <typename Model, class Scalar = double, int N_PARAMETERS = duna::Dynamic, int N_MODEL_OUTPUTS = duna::Dynamic>
    class CostFunctionAnalyticalCovariance : public CostFunctionBase<Scalar>
    {
    public:
        using ParameterVector = Eigen::Matrix<Scalar, N_PARAMETERS, 1>;
        using HessianMatrix = Eigen::Matrix<Scalar, N_PARAMETERS, N_PARAMETERS>;
        using JacobianMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, N_PARAMETERS, Eigen::RowMajor>;
        using ResidualVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

        // TODO change pointer to smartpointer
        CostFunctionAnalyticalCovariance(Model *model, int num_residuals, bool delete_model = false) : m_model(model), m_delete_model(delete_model),
                                                                                                       CostFunctionBase<Scalar>(num_residuals, N_MODEL_OUTPUTS), covariance_inverse_(1)
        {
            init();
        }

        CostFunctionAnalyticalCovariance(Model *model, bool delete_model = false) : m_model(model), m_delete_model(delete_model),
                                                                                    CostFunctionBase<Scalar>(1, N_MODEL_OUTPUTS), covariance_inverse_(1)
        {
            init();
        }

        CostFunctionAnalyticalCovariance(const CostFunctionAnalyticalCovariance &) = delete;
        CostFunctionAnalyticalCovariance &operator=(const CostFunctionAnalyticalCovariance &) = delete;

        ~CostFunctionAnalyticalCovariance()
        {
            if (m_delete_model)
                delete m_model;
        }

        Scalar computeCost(const Scalar *x, bool setup_data) override
        {
            Scalar sum = 0;
            residuals_.resize(m_num_residuals * N_MODEL_OUTPUTS);

            // TODO make dirty variables?
            if (setup_data)
                m_model->setup(x);

            for (int i = 0; i < m_num_residuals; ++i)
            {
                (*m_model)(x, residuals_.template block<N_MODEL_OUTPUTS, 1>(i * N_MODEL_OUTPUTS, 0).data(), i);
            }
            sum = 2 * residuals_.transpose() * covariance_inverse_ * residuals_;
            return sum;
        }

        Scalar linearize(const Scalar *x, Scalar *hessian, Scalar *b) override
        {
            Eigen::Map<HessianMatrix> hessian_map(hessian);
            Eigen::Map<ParameterVector> b_map(b);

            jacobian_.resize(m_num_residuals * N_MODEL_OUTPUTS, N_PARAMETERS);
            residuals_.resize(m_num_residuals * N_MODEL_OUTPUTS);

            hessian_map.setZero();
            b_map.setZero();

            Scalar sum = 0.0;

            (*m_model).setup(x);
            for (int i = 0; i < m_num_residuals; ++i)
            {
                (*m_model)(x, residuals_.template block<N_MODEL_OUTPUTS, 1>(i * N_MODEL_OUTPUTS, 0).data(), i);
                (*m_model).df(x, jacobian_.template block<N_MODEL_OUTPUTS, N_PARAMETERS>(i * N_MODEL_OUTPUTS, 0).data(), i);
            }
            hessian_map.noalias() = jacobian_.transpose() * covariance_inverse_ * jacobian_;
            hessian_ = hessian_map;// coppies hessian locally
            // hessian_map.template selfadjointView<Eigen::Lower>().rankUpdate(jacobian_.transpose()); // H = J^T * J
            // hessian_map.template triangularView<Eigen::Upper>() = hessian_map.transpose();
            b_map.noalias() = jacobian_.transpose() * covariance_inverse_ * residuals_;
            sum = 2 * residuals_.transpose() * covariance_inverse_ * residuals_;
            return sum;
        }

        // Set (scalar) measurement covariance.
        void setCovariance(const Scalar covariance)
        {
            covariance_inverse_ = 1 / covariance;
        }

        // Gets computed P_k = (I - KH)P_k-1
        HessianMatrix computeUpdatedCovariance(const HessianMatrix &P) const
        {
            HessianMatrix kalman_gain = computeUpdatedKalmanGain(P);

            return (HessianMatrix::Identity() - kalman_gain * jacobian_) * P;
        }

    protected:
        Model *m_model;
        // Holds results for cost computations
        using CostFunctionBase<Scalar>::m_num_outputs;
        using CostFunctionBase<Scalar>::m_num_residuals;
        JacobianMatrix jacobian_;
        ResidualVector residuals_;
        HessianMatrix hessian_; // local hessian

        // Covariance
        Scalar covariance_inverse_;

        bool m_delete_model;

        void init()
        {
            static_assert(N_PARAMETERS != -1, "Dynamic Cost Function not yet implemented");
        }

    private:
        // Computes K = (H^T * R^(-1) * H + P^(-1) )^-1 H^T * R^(-1) with a given state covariance P.
        //      Hessian <---------------|   |
        //      State Cov                <--|
        //         
        HessianMatrix computeUpdatedKalmanGain(const HessianMatrix &P) const
        {
            return (hessian_ + P.inverse()).inverse() * jacobian_.transpose() * covariance_inverse_;
        }
    };
}