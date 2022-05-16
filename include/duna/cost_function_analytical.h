#ifndef COSTFUNCTIONANALYTICAL_H
#define COSTFUNCTIONANALYTICAL_H

#include <duna/cost_function.h>

namespace duna
{
    // NOTE. We are using Model as a template to be able to call its copy constructors and enable numercial diff.
    template <typename Model, class Scalar = double, int N_PARAMETERS = duna::Dynamic, int N_MODEL_OUTPUTS = duna::Dynamic>
    class CostFunctionAnalytical : public CostFunctionBase<Scalar>
    {
    public:
        using ParameterVector = Eigen::Matrix<Scalar, N_PARAMETERS, 1>;
        using ResidualVector = Eigen::Matrix<Scalar, N_MODEL_OUTPUTS, 1>;
        using HessianMatrix = Eigen::Matrix<Scalar, N_PARAMETERS, N_PARAMETERS>;
        using JacobianMatrix = Eigen::Matrix<Scalar, N_MODEL_OUTPUTS, N_PARAMETERS>;

        // TODO change pointer to smartpointer
        CostFunctionAnalytical(Model *model, int num_residuals) : m_model(model),
                                                                  residuals(nullptr),
                                                                  jacobian(nullptr),
                                                                  CostFunctionBase<Scalar>(num_residuals,N_MODEL_OUTPUTS)
        {
            init();
        }

        CostFunctionAnalytical(Model *model) : m_model(model),
                                               residuals(nullptr),
                                               jacobian(nullptr),
                                               CostFunctionBase<Scalar>(1,N_MODEL_OUTPUTS)
        {
            init();
            // TODO remove warning
            std::cout << "Warning, num_residuals not set\n";
        }

        CostFunctionAnalytical(const CostFunctionAnalytical &) = delete;
        CostFunctionAnalytical &operator=(const CostFunctionAnalytical &) = delete;

        ~CostFunctionAnalytical()
        {
            delete[] residuals_data;
            delete[] jacobian_data;
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

            Scalar sum = 0.0;

            // Step size
            Scalar *h = new Scalar[x_map.size()];

            m_model[0].setup(x_map.data()); // this was inside the loop below.. Very bad.

            for (int i = 0; i < m_num_residuals; ++i)
            {

                m_model[0](x_map.data(), residuals_data, i);
                sum += 2 * residuals.squaredNorm();

                m_model[0].df(x_map.data(), jacobian_data, i);

                // hessian.template selfadjointView<Eigen::Lower>().rankUpdate(jacobian_row.transpose()); // this sums ? yes
                hessian_map.noalias() += (jacobian.transpose() * jacobian);
                b_map.noalias() += jacobian.transpose() * residuals;
            }

            delete h;

            hessian_map.template triangularView<Eigen::Upper>() = hessian_map.transpose();

            return sum;
        }

    protected:
        Model *m_model;
        // Holds results for cost computations
        Scalar *residuals_data;
        Scalar *jacobian_data;

        Eigen::Map<const ResidualVector> residuals;
        Eigen::Map<const JacobianMatrix> jacobian;

        using CostFunctionBase<Scalar>::m_num_outputs;
        using CostFunctionBase<Scalar>::m_num_residuals;

        void init()
        {
            // m_num_outputs = N_MODEL_OUTPUTS;
            residuals_data = new Scalar[m_num_outputs];
            jacobian_data = new Scalar[m_num_outputs * N_PARAMETERS];

            // Map allocated arrays to eigen types using placement new syntax.
            new (&residuals) Eigen::Map<const ResidualVector>(residuals_data);
            new (&jacobian) Eigen::Map<const JacobianMatrix>(jacobian_data);

            static_assert(N_PARAMETERS != -1, "Dynamic Cost Function not yet implemented");
        }
    };
}

#endif