#ifndef COSTFUNCTIONNUMERICAL_H
#define COSTFUNCTIONNUMERICAL_H

#include <duna/cost_function.h>
#include <duna/model.h>
#include <duna/logger.h>

namespace duna
{
    /* Numerical Differentiation cost function module. Computes numerical derivatives when computing hessian. */
    template <class Scalar = double, int N_PARAMETERS = duna::Dynamic, int N_MODEL_OUTPUTS = duna::Dynamic>
    class CostFunctionNumerical : public CostFunctionBase<Scalar>
    {
    public:
        using ParameterVector = Eigen::Matrix<Scalar, N_PARAMETERS, 1>;
        using HessianMatrix = Eigen::Matrix<Scalar, N_PARAMETERS, N_PARAMETERS>;
        using JacobianMatrix = Eigen::Matrix<Scalar, N_MODEL_OUTPUTS, N_PARAMETERS, Eigen::RowMajor>;
        using ResidualVector = Eigen::Matrix<Scalar, N_MODEL_OUTPUTS, 1>;
        using typename CostFunctionBase<Scalar>::Model;
        using typename CostFunctionBase<Scalar>::ModelPtr;

        CostFunctionNumerical(ModelPtr model, int num_residuals) : CostFunctionBase<Scalar>(model, num_residuals),
                                                                       hessian_map_(0, 0, 0), x_map_(0, 0, 0), b_map_(0, 0, 0)
        {
        }
        CostFunctionNumerical(ModelPtr model) : CostFunctionBase<Scalar>(model, 1),
                                                    hessian_map_(0, 0, 0), x_map_(0, 0, 0), b_map_(0, 0, 0)
        {
        }
        CostFunctionNumerical(const CostFunctionNumerical &) = delete;
        CostFunctionNumerical &operator=(const CostFunctionNumerical &) = delete;

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

        virtual Scalar linearize(const Scalar *x, Scalar *hessian, Scalar *b) override
        {
            init(x, hessian, b);

            Scalar sum = 0.0;

            const Scalar min_step_size = std::sqrt(std::numeric_limits<Scalar>::epsilon());

            // Step size
            std::vector<Scalar> h(x_map_.size());

            // TODO check if at least a few residuals were computed.
            // TODO Optimize!!
            for (int i = 0; i < m_num_residuals; ++i)
            {
                model_->setup(x_map_.data());

                if (model_->f(x, residuals_.data(), i))
                {
                    std::vector<ParameterVector> x_plus(x_map_.size(), x_map_);
                    for (int j = 0; j < x_map_.size(); ++j)
                    {
                        h[j] = min_step_size * abs(x_map_[j]);

                        if (h[j] == 0.0)
                            h[j] = min_step_size;

                        x_plus[j][j] += h[j];

                        model_->setup((x_plus[j]).data());
                        model_->f(x_plus[j].data(), residuals_plus_.data(), i);

                        jacobian_.col(j) = (residuals_plus_ - residuals_) / h[j];
                    }

                    Scalar w = loss_function_->weight(residuals_.squaredNorm());
                    // hessian_map.template selfadjointView<Eigen::Lower>().rankUpdate(jacobian_.transpose()); // H = J^T * J
                    hessian_map_.noalias() += jacobian_.transpose() * w * jacobian_;
                    b_map_.noalias() += jacobian_.transpose() * w * residuals_;
                    sum += residuals_.transpose() * residuals_;
                }
            }
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
        ResidualVector residuals_plus_;

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

#endif