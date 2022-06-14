#ifndef COSTFUNCTIONANALYTICAL_H
#define COSTFUNCTIONANALYTICAL_H

#include <duna/cost_function.h>
#include <duna/logging.h>

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
        using JacobianBlockMatrix = Eigen::Matrix<Scalar, N_MODEL_OUTPUTS, N_PARAMETERS>;
        using JacobianMatrix = Eigen::Matrix<Scalar, 1, N_PARAMETERS>;

        // TODO change pointer to smartpointer
        CostFunctionAnalytical(Model *model, int num_residuals, bool delete_model = false) : m_model(model), m_delete_model(delete_model),
                                                                                             CostFunctionBase<Scalar>(num_residuals, N_MODEL_OUTPUTS)
        {
            init();
        }

        CostFunctionAnalytical(Model *model, bool delete_model = false) : m_model(model), m_delete_model(delete_model),
                                                                          CostFunctionBase<Scalar>(1, N_MODEL_OUTPUTS)
        {

            init();
            // TODO remove warning
            DUNA_DEBUG("Warning, num_residuals not set\n");
        }

        CostFunctionAnalytical(const CostFunctionAnalytical &) = delete;
        CostFunctionAnalytical &operator=(const CostFunctionAnalytical &) = delete;

        ~CostFunctionAnalytical()
        {
            if (m_delete_model)
                delete m_model;
        }

        Scalar computeCost(const Scalar *x, bool setup_data) override
        {
            Scalar sum = 0;
            ResidualVector residuals;

            // TODO make dirty variables?
            if (setup_data)
                m_model->setup(x);

            for (int i = 0; i < m_num_residuals; ++i)
            {
                (*m_model)(x, residuals.data(), i);
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

            m_model[0].setup(x_map.data()); // this was inside the loop below.. Very bad.

            ResidualVector residuals;
            JacobianBlockMatrix jacobian_row;

            for (int i = 0; i < m_num_residuals; ++i)
            {

                m_model[0](x_map.data(), residuals.data(), i);
                sum += 2 * residuals.squaredNorm();

                m_model[0].df(x_map.data(), jacobian_row.data(), i);

                hessian_map.template selfadjointView<Eigen::Lower>().rankUpdate(jacobian_row.transpose()); // this sums ? yes
                // hessian_map.noalias() += (jacobian_row.transpose() * jacobian_row);
                b_map.noalias() += jacobian_row.transpose() * residuals;
            }

            hessian_map.template triangularView<Eigen::Upper>() = hessian_map.transpose();

            return sum;
        }

    protected:
        Model *m_model;
        // Holds results for cost computations
        using CostFunctionBase<Scalar>::m_num_outputs;
        using CostFunctionBase<Scalar>::m_num_residuals;
        bool m_delete_model;

        void init()
        {
            static_assert(N_PARAMETERS != -1, "Dynamic Cost Function not yet implemented");
        }
    };
}

#endif