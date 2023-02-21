#pragma once

#include <duna/cost_function.h>
#include <duna/model.h>
#include <duna/logger.h>
#include <memory>

namespace duna
{
    /* Analytical cost function module. Computes hessian using provided `f_df` model function with explicit jacobian calculation. */
    template <class Scalar = double, int N_PARAMETERS = duna::Dynamic, int N_MODEL_OUTPUTS = duna::Dynamic>
    class CostFunctionAnalytical : public CostFunctionBase<Scalar>
    {
    public:
        using ParameterVector = Eigen::Matrix<Scalar, N_PARAMETERS, 1>;
        using HessianMatrix = Eigen::Matrix<Scalar, N_PARAMETERS, N_PARAMETERS>;
        using JacobianMatrix = Eigen::Matrix<Scalar, N_MODEL_OUTPUTS, N_PARAMETERS, Eigen::RowMajor>;
        using ResidualVector = Eigen::Matrix<Scalar, N_MODEL_OUTPUTS, 1>;
        using typename CostFunctionBase<Scalar>::Model;
        using typename CostFunctionBase<Scalar>::ModelPtr;

        // TODO change pointer to smartpointer
        CostFunctionAnalytical(ModelPtr model, int num_residuals) : CostFunctionBase<Scalar>(model, num_residuals),
                                                                    hessian_map_(0, 0, 0), x_map_(0, 0, 0), b_map_(0, 0, 0)
        {
        }
        CostFunctionAnalytical(ModelPtr model) : CostFunctionBase<Scalar>(model, 1),
                                                 hessian_map_(0, 0, 0), x_map_(0, 0, 0), b_map_(0, 0, 0)
        {
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

            init(x, hessian, b);

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
                    // std::cout << "residuals_" << residuals_ << std::endl;
                    // hessian_map.template selfadjointView<Eigen::Lower>().rankUpdate(jacobian_.transpose()); // H = J^T * J
                    hessian_map_.noalias() += jacobian_.transpose() * w * jacobian_;
                    b_map_.noalias() += jacobian_.transpose() * w * residuals_;
                    sum += residuals_.transpose() * residuals_;
                }
            }
            // std::cout << "hessian_map_:\n " << hessian_map_ << std::endl;
            hessian_map_.template triangularView<Eigen::Upper>() = hessian_map_.transpose();
            return sum;
        }

    protected:
        using CostFunctionBase<Scalar>::m_num_residuals;
        using CostFunctionBase<Scalar>::model_;
        using CostFunctionBase<Scalar>::loss_function_;
        Eigen::Map<const ParameterVector> x_map_;
        Eigen::Map<HessianMatrix> hessian_map_;
        Eigen::Map<ParameterVector> b_map_;

        JacobianMatrix jacobian_;
        ResidualVector residuals_;

        // Initialize internal cost function states.
        virtual void init(const Scalar *x, Scalar *hessian, Scalar *b) override
        {
            new (&x_map_) Eigen::Map<const ParameterVector>(x, N_PARAMETERS, 1);
            new (&hessian_map_) Eigen::Map<HessianMatrix>(hessian, N_PARAMETERS, N_PARAMETERS);
            new (&b_map_) Eigen::Map<ParameterVector>(b, N_PARAMETERS, 1);

            hessian_map_.setZero();
            b_map_.setZero();
        }
    };
}