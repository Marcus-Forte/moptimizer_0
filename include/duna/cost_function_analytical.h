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
        using JacobianMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, N_PARAMETERS, Eigen::RowMajor>;
        using ResidualVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
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
            residuals_.resize(m_num_residuals * N_MODEL_OUTPUTS);

            model_->setup(x);

            int valid_errors = 0;

            for (int i = 0; i < m_num_residuals; ++i)
            {
                if (model_->f(x, residuals_.template block<N_MODEL_OUTPUTS, 1>(valid_errors * N_MODEL_OUTPUTS, 0).data(), i))
                    valid_errors++;
            }

            ResidualVector valid_residuals = residuals_.block(0, 0, valid_errors * N_MODEL_OUTPUTS, 1);

            sum = valid_residuals.transpose() * valid_residuals;
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

            model_->setup(x);

            int valid_errors = 0;

            for (int i = 0; i < m_num_residuals; ++i)
            {
                if (model_->f_df(x,
                                 residuals_.template block<N_MODEL_OUTPUTS, 1>(valid_errors * N_MODEL_OUTPUTS, 0).data(),
                                 jacobian_.template block<N_MODEL_OUTPUTS, N_PARAMETERS>(valid_errors * N_MODEL_OUTPUTS, 0).data(),
                                 i))
                {
                    valid_errors++;
                }
            }

            // Select only valid residues.
            JacobianMatrix &&valid_jacobian = jacobian_.block(0, 0, valid_errors * N_MODEL_OUTPUTS, N_PARAMETERS);
            ResidualVector &&valid_residuals = residuals_.block(0, 0, valid_errors * N_MODEL_OUTPUTS, 1);

            // std::cout << valid_jacobian << std::endl;

            hessian_map.template selfadjointView<Eigen::Lower>().rankUpdate(valid_jacobian.transpose()); // H = J^T * J
            hessian_map.template triangularView<Eigen::Upper>() = hessian_map.transpose();
            b_map.noalias() = valid_jacobian.transpose() * valid_residuals;
            sum =  valid_residuals.transpose() * valid_residuals;
            return sum;
        }

    protected:
        using CostFunctionBase<Scalar>::m_num_outputs;
        using CostFunctionBase<Scalar>::m_num_residuals;
        using CostFunctionBase<Scalar>::model_;
        JacobianMatrix jacobian_;
        ResidualVector residuals_;

        void init()
        {
            static_assert(N_PARAMETERS != -1, "Dynamic Cost Function not yet implemented");
        }
    };
}