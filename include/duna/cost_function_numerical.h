#ifndef COSTFUNCTIONNUMERICAL_H
#define COSTFUNCTIONNUMERICAL_H

#include <duna/cost_function.h>

namespace duna
{
    // NOTE. We are using Model as a template to be able to call its copy constructors and enable numercial diff.
    template <typename Model, class Scalar = double, int N_PARAMETERS = duna::Dynamic, int N_MODEL_OUTPUTS = duna::Dynamic>
    class CostFunctionNumericalDiff : public CostFunctionBase<Scalar>
    {
    public:
        using ParameterVector = Eigen::Matrix<Scalar, N_PARAMETERS, 1>;
        using ResidualVector = Eigen::Matrix<Scalar, N_MODEL_OUTPUTS, 1>;
        using HessianMatrix = Eigen::Matrix<Scalar, N_PARAMETERS, N_PARAMETERS>;
        using JacobianMatrix = Eigen::Matrix<Scalar, N_MODEL_OUTPUTS, N_PARAMETERS>;

        // TODO change pointer to smartpointer
        CostFunctionNumericalDiff(Model *model, int num_residuals) : m_model(model),
                                                                     residuals(nullptr),
                                                                     residuals_plus(nullptr),
                                                                     residuals_minus(nullptr),
                                                                     CostFunctionBase<Scalar>(num_residuals, N_MODEL_OUTPUTS)
        {
            init();
        }

        CostFunctionNumericalDiff(Model *model) : m_model(model),
                                                  residuals(nullptr),
                                                  residuals_plus(nullptr),
                                                  residuals_minus(nullptr),
                                                  CostFunctionBase<Scalar>(1, N_MODEL_OUTPUTS)
        {
            init();
            // TODO remove warning
            std::cout << "Warning, num_residuals not set\n";
        }

        CostFunctionNumericalDiff(const CostFunctionNumericalDiff &) = delete;
        CostFunctionNumericalDiff &operator=(const CostFunctionNumericalDiff &) = delete;

        ~CostFunctionNumericalDiff()
        {
            delete[] residuals_data;
            delete[] residuals_plus_data;
            delete[] residuals_minus_data;
        }

        Scalar computeCost(const Scalar *x, bool setup_data) override
        {
            Scalar sum = 0;

            // TODO make dirty variables?
            if (setup_data)
                m_model->setup(x);

            for (int i = 0; i < m_num_residuals; ++i)
            {
                (*m_model)(x, residuals_data, i);
                sum += 2 * residuals.squaredNorm();
            }

            return sum;
        }

        Scalar linearize(const Scalar *x, Scalar *hessian, Scalar *b) override
        {
            Eigen::Map<const ParameterVector> x_map(x);
            Eigen::Map<HessianMatrix> hessian_map(hessian);
            Eigen::Map<ParameterVector> b_map(b);

            hessian_map.setZero();
            b_map.setZero();

            JacobianMatrix jacobian_row;
            Scalar sum = 0.0;

            const Scalar min_step_size = std::sqrt(std::numeric_limits<Scalar>::epsilon());
            // const Scalar min_step_size = 12 * std::numeric_limits<Scalar>::epsilon();
            // const Scalar min_step_size = 0.0001;

            // Create a new model for each numerical increment
            std::vector<Model> diff_plus(x_map.size(), *m_model);
            // std::vector<Model> diff_minus(x0.size(), *m_model);

            std::vector<ParameterVector> x_plus(x_map.size(), x_map);
            // std::vector<ParameterVector> x_minus(x0.size(), x0);

            // Step size
            Scalar *h = new Scalar[x_map.size()];

            for (int j = 0; j < x_map.size(); ++j)
            {
                h[j] = min_step_size * abs(x_map[j]);

                if (h[j] == 0.0)
                    h[j] = min_step_size;

                // TODO Manifold operation
                x_plus[j][j] += h[j];
                // x_minus[j][j] -= h[j];

                diff_plus[j].setup((x_plus[j]).data());
                // diff_minus[j].setup((x_minus[j]).data());
            }

            m_model[0].setup(x_map.data()); // this was inside the loop below.. Very bad.

            for (int i = 0; i < m_num_residuals; ++i)
            {

                m_model[0](x_map.data(), residuals_data, i);
                sum += 2 * residuals.squaredNorm();

                for (int j = 0; j < x_map.size(); ++j)
                {
                    diff_plus[j](x_plus[j].data(), residuals_plus_data, i);
                    // diff_minus[j](x_minus[j].data(), residuals_minus_data, i);

                    jacobian_row.col(j) = (residuals_plus - residuals) / (1 * h[j]);
                }

                // hessian.template selfadjointView<Eigen::Lower>().rankUpdate(jacobian_row.transpose()); // this sums ? yes
                hessian_map.noalias() += (jacobian_row.transpose() * jacobian_row);
                b_map.noalias() += jacobian_row.transpose() * residuals;
            }

            delete h;

            hessian_map.template triangularView<Eigen::Upper>() = hessian_map.transpose();

            return sum;
        }

    protected:
        Model *m_model;
        // Holds results for cost computations
        Scalar *residuals_data;
        Scalar *residuals_plus_data;
        Scalar *residuals_minus_data;
        Eigen::Map<const ResidualVector> residuals;
        Eigen::Map<const ResidualVector> residuals_plus;
        Eigen::Map<const ResidualVector> residuals_minus;

        using CostFunctionBase<Scalar>::m_num_outputs;
        using CostFunctionBase<Scalar>::m_num_residuals;

        void init()
        {
            // m_num_outputs = N_MODEL_OUTPUTS;
            residuals_data = new Scalar[m_num_outputs];
            residuals_plus_data = new Scalar[m_num_outputs];
            residuals_minus_data = new Scalar[m_num_outputs];

            // Map allocated arrays to eigen types using placement new syntax.
            new (&residuals) Eigen::Map<const ResidualVector>(residuals_data);
            new (&residuals_plus) Eigen::Map<const ResidualVector>(residuals_plus_data);
            new (&residuals_minus) Eigen::Map<const ResidualVector>(residuals_minus_data);

            static_assert(N_PARAMETERS != -1, "Dynamic Cost Function not yet implemented");
        }
    };
}

#endif