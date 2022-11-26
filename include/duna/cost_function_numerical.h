#ifndef COSTFUNCTIONNUMERICAL_H
#define COSTFUNCTIONNUMERICAL_H

#include <duna/cost_function.h>
#include <duna/model.h>
#include <duna/logger.h>

namespace duna
{
    // NOTE. We are using Model as a template to be able to call its copy constructors and enable numercial diff.
    template <class Scalar = double, int N_PARAMETERS = duna::Dynamic, int N_MODEL_OUTPUTS = duna::Dynamic>
    class CostFunctionNumericalDiff : public CostFunctionBase<Scalar>
    {
    public:
        using ParameterVector = Eigen::Matrix<Scalar, N_PARAMETERS, 1>;
        using HessianMatrix = Eigen::Matrix<Scalar, N_PARAMETERS, N_PARAMETERS>;
        using JacobianMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, N_PARAMETERS, Eigen::RowMajor>;
        using ResidualVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        using typename CostFunctionBase<Scalar>::Model;
        using typename CostFunctionBase<Scalar>::ModelPtr;

        CostFunctionNumericalDiff(ModelPtr model, int num_residuals) : CostFunctionBase<Scalar>(model, num_residuals, N_MODEL_OUTPUTS)
        {
            init();
        }

        CostFunctionNumericalDiff(ModelPtr model) : CostFunctionBase<Scalar>(model, 1, N_MODEL_OUTPUTS)
        {
            init();
        }

        CostFunctionNumericalDiff(const CostFunctionNumericalDiff &) = delete;
        CostFunctionNumericalDiff &operator=(const CostFunctionNumericalDiff &) = delete;

        void update(const Scalar *x) override
        {
            model_->update(x);
        }

        Scalar computeCost(const Scalar *x) override
        {
            Scalar sum = 0;
            residuals_.resize(m_num_residuals * N_MODEL_OUTPUTS);

            model_->setup(x);

            for (int i = 0; i < m_num_residuals; ++i)
            {
                (*model_)(x, residuals_.template block<N_MODEL_OUTPUTS, 1>(i * N_MODEL_OUTPUTS, 0).data(), i);
            }
            sum = 2 * residuals_.transpose() * residuals_;
            return sum;
        }

        virtual Scalar linearize(const Scalar *x, Scalar *hessian, Scalar *b) override
        {
            Eigen::Map<const ParameterVector> x_map(x);
            Eigen::Map<HessianMatrix> hessian_map(hessian);
            Eigen::Map<ParameterVector> b_map(b);

            jacobian_.resize(m_num_residuals * N_MODEL_OUTPUTS, N_PARAMETERS);
            residuals_.resize(m_num_residuals * N_MODEL_OUTPUTS);
            residuals_plus_.resize(m_num_residuals * N_MODEL_OUTPUTS);

            hessian_map.setZero();
            b_map.setZero();

            Scalar sum = 0.0;

            const Scalar min_step_size = std::sqrt(std::numeric_limits<Scalar>::epsilon());

            // std::vector<Model> diff_minus(x_map.size(), *m_model);

            std::vector<ParameterVector> x_plus(x_map.size(), x_map);
            // std::vector<ParameterVector> x_minus(x_map.size(), x_map);

            // Step size
            std::vector<Scalar> h(x_map.size());

            model_->setup(x_map.data());

            for (int i = 0; i < m_num_residuals; ++i)
            {
                (*model_)(x, residuals_.template block<N_MODEL_OUTPUTS, 1>(i * N_MODEL_OUTPUTS, 0).data(), i);
            }

            for (int j = 0; j < x_map.size(); ++j)
            {

                h[j] = min_step_size * abs(x_map[j]);

                if (h[j] == 0.0)
                    h[j] = min_step_size;

                // TODO Manifold operation
                x_plus[j][j] += h[j];
                // x_minus[j][j] -= h[j];

                model_->setup((x_plus[j]).data());

                for (int i = 0; i < m_num_residuals; ++i)
                {
                    (*model_)(x_plus[j].data(), residuals_plus_.template block<N_MODEL_OUTPUTS, 1>(i * N_MODEL_OUTPUTS, 0).data(), i);
                }
                jacobian_.col(j) = (residuals_plus_ - residuals_) / h[j];
            }

            hessian_map.template selfadjointView<Eigen::Lower>().rankUpdate(jacobian_.transpose()); // H = J^T * J
            hessian_map.template triangularView<Eigen::Upper>() = hessian_map.transpose();
            b_map.noalias() = jacobian_.transpose() * residuals_;
            sum = 2 * residuals_.transpose() * residuals_;
            return sum;
        }

    private:
        using CostFunctionBase<Scalar>::m_num_outputs;
        using CostFunctionBase<Scalar>::m_num_residuals;
        using CostFunctionBase<Scalar>::model_;
        JacobianMatrix jacobian_;
        ResidualVector residuals_;
        ResidualVector residuals_plus_;

        // TODO test if dynamic
        void init()
        {
            static_assert(N_PARAMETERS != -1, "Dynamic Cost Function not yet implemented");
        }
    };
}

#endif