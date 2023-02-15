#pragma once

#include <duna/cost_function.h>
#include <duna/model.h>
#include <duna/logger.h>
#include <memory>

namespace duna
{
    // NOTE. We are using Model as a template to be able to call its copy constructors and enable numercial diff.
    template <class Scalar = double, int N_PARAMETERS = duna::Dynamic, int N_MODEL_OUTPUTS = duna::Dynamic>
    class CostFunctionAnalytical : public CostFunctionBase<Scalar>
    {
    public:
        using ParameterVector = Eigen::Matrix<Scalar, N_PARAMETERS, 1>;
        using HessianMatrix = Eigen::Matrix<Scalar, N_PARAMETERS, N_PARAMETERS>;
        using JacobianMatrix = Eigen::Matrix<Scalar, N_MODEL_OUTPUTS, N_PARAMETERS, Eigen::RowMajor>;
        using ResidualVector = Eigen::Matrix<Scalar, N_MODEL_OUTPUTS, 1>;
        using Model = BaseModelJacobian<Scalar>;
        using ModelPtr = typename BaseModelJacobian<Scalar>::Ptr;

        // TODO change pointer to smartpointer
        CostFunctionAnalytical(ModelPtr model, int num_residuals) : CostFunctionBase<Scalar>(model, num_residuals, N_MODEL_OUTPUTS)
        {
            init();
        }

        CostFunctionAnalytical(ModelPtr model) : CostFunctionBase<Scalar>(model, 1, N_MODEL_OUTPUTS)
        {
            init();
        }

        CostFunctionAnalytical(const CostFunctionAnalytical &) = delete;
        CostFunctionAnalytical &operator=(const CostFunctionAnalytical &) = delete;

        Scalar computeCost(const Scalar *x) override
        {
            Scalar sum = 0;

            model_->setup(x);

            for (int i = 0; i < m_num_residuals; ++i)
            {
                if (model_->f(x, residuals_.data(), i))
                {
                    sum += residuals_.transpose() * residuals_;
                }
            }
            return sum;
        }

        Scalar linearize(const Scalar *x, Scalar *hessian, Scalar *b) override
        {
            Eigen::Map<HessianMatrix> hessian_map(hessian);
            Eigen::Map<ParameterVector> b_map(b);

            hessian_map.setZero();
            b_map.setZero();

            Scalar sum = 0.0;

            model_->setup(x);

            // TODO check if at least a few residuals were computed.
            for (int i = 0; i < m_num_residuals; ++i)
            {
                if (model_->f_df(x,
                                 residuals_.data(),
                                 jacobian_.data(),
                                 i))
                {
                    Scalar w = loss_function_->weight(residuals_.squaredNorm());
                    // hessian_map.template selfadjointView<Eigen::Lower>().rankUpdate(jacobian_.transpose()); // H = J^T * J
                    hessian_map.noalias() += jacobian_.transpose() * w * jacobian_;
                    b_map.noalias() += jacobian_.transpose() * w * residuals_;
                    sum += residuals_.transpose() * residuals_;
                }
            }
            hessian_map.template triangularView<Eigen::Upper>() = hessian_map.transpose();
            return sum;
        }

    protected:
        using CostFunctionBase<Scalar>::m_num_outputs;
        using CostFunctionBase<Scalar>::m_num_residuals;
        using CostFunctionBase<Scalar>::model_;
        using CostFunctionBase<Scalar>::loss_function_;
        JacobianMatrix jacobian_;
        ResidualVector residuals_;

        void init()
        {
            static_assert(N_PARAMETERS != -1, "Dynamic Cost Function not yet implemented");
        }
    };
}