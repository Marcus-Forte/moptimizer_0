#ifndef COSTFUNCTIONANALYTICAL_H
#define COSTFUNCTIONANALYTICAL_H

#include <duna/cost_function.h>
#include <duna/logger.h>

namespace duna
{
    // NOTE. We are using Model as a template to be able to call its copy constructors and enable numercial diff.
    template <typename Model, class Scalar = double, int N_PARAMETERS = duna::Dynamic, int N_MODEL_OUTPUTS = duna::Dynamic>
    class CostFunctionAnalytical : public CostFunctionBase<Scalar>
    {
    public:
        using ParameterVector = Eigen::Matrix<Scalar, N_PARAMETERS, 1>;
        using HessianMatrix = Eigen::Matrix<Scalar, N_PARAMETERS, N_PARAMETERS>;
        using JacobianMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, N_PARAMETERS, Eigen::RowMajor>;
        using ResidualVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

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
            residuals_.resize(m_num_residuals * N_MODEL_OUTPUTS);

            // TODO make dirty variables?
            if (setup_data)
                m_model->setup(x);

            for (int i = 0; i < m_num_residuals; ++i)
            {
                (*m_model)(x, residuals_.template block<N_MODEL_OUTPUTS, 1>(i * N_MODEL_OUTPUTS, 0).data(), i);
            }
            sum = 2 * residuals_.transpose() * residuals_;
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
            // hessian_map.noalias() = jacobian_.transpose() * jacobian_;
            hessian_map.template selfadjointView<Eigen::Lower>().rankUpdate(jacobian_.transpose()); // H = J^T * J
            hessian_map.template triangularView<Eigen::Upper>() = hessian_map.transpose();
            b_map.noalias() = jacobian_.transpose() * residuals_;
            sum = 2 * residuals_.transpose() * residuals_;
            return sum;
        }

    protected:
        Model *m_model;
        // Holds results for cost computations
        using CostFunctionBase<Scalar>::m_num_outputs;
        using CostFunctionBase<Scalar>::m_num_residuals;
        JacobianMatrix jacobian_;
        ResidualVector residuals_;

        bool m_delete_model;

        void init()
        {
            static_assert(N_PARAMETERS != -1, "Dynamic Cost Function not yet implemented");
        }
    };
}

#endif