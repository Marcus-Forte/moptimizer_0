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
                                                                     CostFunctionBase<Scalar>(num_residuals, N_MODEL_OUTPUTS)
        {
            init();
        }

        CostFunctionNumericalDiff(Model *model) : m_model(model),
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
        }

        Scalar computeCost(const Scalar *x, bool setup_data) override
        {
            Scalar sum = 0;
            ResidualVector residuals;

            // TODO make dirty variables?
            if (setup_data)
                m_model->setup(x);

#pragma omp parallel num_threads(m_num_threads)
            {
                ResidualVector residuals;
#pragma omp for reduction(+ \
                          : sum) schedule(guided, 8)
                for (int i = 0; i < m_num_residuals; ++i)
                {
                    (*m_model)(x, residuals.data(), i);
                    sum += 2 * residuals.squaredNorm();
                }
            }

            return sum;
        }

        Scalar linearize(const Scalar *x, Scalar *hessian, Scalar *b) override
        {
            Eigen::Map<const ParameterVector> x_map(x);

            Scalar sum = 0.0;

            const Scalar min_step_size = std::sqrt(std::numeric_limits<Scalar>::epsilon());
            // const Scalar min_step_size = 12 * std::numeric_limits<Scalar>::epsilon();
            // const Scalar min_step_size = 0.0001;

            // Create a new model for each numerical increment
            std::vector<Model> diff_plus(x_map.size(), *m_model);
            // std::vector<Model> diff_minus(x_map.size(), *m_model);

            std::vector<ParameterVector> x_plus(x_map.size(), x_map);
            // std::vector<ParameterVector> x_minus(x_map.size(), x_map);

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

            std::vector<HessianMatrix> Hs(m_num_threads);
            std::vector<ParameterVector> Bs(m_num_threads);
            for (int i = 0; i < m_num_threads; ++i)
            {
                Hs[i].setZero();
                Bs[i].setZero();
            }

#pragma omp parallel num_threads(m_num_threads)
            {
                ResidualVector residuals;
                ResidualVector residuals_plus;
                JacobianMatrix jacobian_row;
#pragma omp for reduction(+ \
                          : sum) schedule(guided, 8)
                for (int i = 0; i < m_num_residuals; ++i)
                {

                    m_model[0](x_map.data(), residuals.data(), i);

                    sum += 2 * residuals.squaredNorm();

                    for (int j = 0; j < x_map.size(); ++j)
                    {
                        diff_plus[j](x_plus[j].data(), residuals_plus.data(), i);
                        // diff_minus[j](x_minus[j].data(), residuals_minus_data, i);

                        jacobian_row.col(j) = (residuals_plus - residuals) / (h[j]);
                    }

                    // hessian.template selfadjointView<Eigen::Lower>().rankUpdate(jacobian_row.transpose()); // this sums ? yes
                    HessianMatrix Hi = (jacobian_row.transpose() * jacobian_row);
                    ParameterVector bi = jacobian_row.transpose() * residuals;

                    Hs[omp_get_thread_num()] += Hi;
                    Bs[omp_get_thread_num()] += bi;

                } // pragma for
            }     // pragma parallel

            Eigen::Map<HessianMatrix> hessian_map(hessian);
            Eigen::Map<ParameterVector> b_map(b);

            hessian_map.setZero();
            b_map.setZero();

            for (int i = 0; i < m_num_threads; ++i)
            {
                hessian_map.noalias() += Hs[i];
                b_map.noalias() += Bs[i];
            }

            hessian_map.template triangularView<Eigen::Upper>() = hessian_map.transpose();

            delete h;
            return sum;
        }

    protected:
        Model *m_model;
        int m_num_threads;
        using CostFunctionBase<Scalar>::m_num_outputs;
        using CostFunctionBase<Scalar>::m_num_residuals;
        // TODO test if dynamic
        void init()
        {
            static_assert(N_PARAMETERS != -1, "Dynamic Cost Function not yet implemented");
            omp_set_num_threads(8);
            m_num_threads = omp_get_max_threads();
        }
    };
}

#endif