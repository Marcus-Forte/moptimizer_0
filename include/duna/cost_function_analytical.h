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
        using Model = BaseModelJacobian<Scalar>;
        using ModelPtr = typename Model::Ptr;
        using ParameterVector = Eigen::Matrix<Scalar, N_PARAMETERS, 1>;
        using HessianMatrix = Eigen::Matrix<Scalar, N_PARAMETERS, N_PARAMETERS>;
        using JacobianMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, N_PARAMETERS, Eigen::RowMajor>;
        using ResidualVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

        // TODO change pointer to smartpointer
        CostFunctionAnalytical(ModelPtr model, int num_residuals) : model_(model),
                                                                    CostFunctionBase<Scalar>(num_residuals, N_MODEL_OUTPUTS)
        {
            init();
        }

        CostFunctionAnalytical(ModelPtr model) : model_(model),
                                                 CostFunctionBase<Scalar>(1, N_MODEL_OUTPUTS)
        {
            init();
        }

        CostFunctionAnalytical(const CostFunctionAnalytical &) = delete;
        CostFunctionAnalytical &operator=(const CostFunctionAnalytical &) = delete;

        void init(const Scalar *x) override
        {
            model_->init(x);
        }

        void setup(const Scalar *x) override
        {
            model_->setup(x);
        }

        Scalar computeCost(const Scalar *x) override
        {

            Scalar sum = 0;
            residuals_.resize(m_num_residuals * N_MODEL_OUTPUTS);

            for (int i = 0; i < m_num_residuals; ++i)
            {
                (*model_)(x, residuals_.template block<N_MODEL_OUTPUTS, 1>(i * N_MODEL_OUTPUTS, 0).data(), i);
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

            for (int i = 0; i < m_num_residuals; ++i)
            {
                (*model_)(x, residuals_.template block<N_MODEL_OUTPUTS, 1>(i * N_MODEL_OUTPUTS, 0).data(), i);
                (*model_).df(x, jacobian_.template block<N_MODEL_OUTPUTS, N_PARAMETERS>(i * N_MODEL_OUTPUTS, 0).data(), i);
            }
            
            // hessian_map.noalias() = jacobian_.transpose() * jacobian_;
            hessian_map.template selfadjointView<Eigen::Lower>().rankUpdate(jacobian_.transpose()); // H = J^T * J
            hessian_map.template triangularView<Eigen::Upper>() = hessian_map.transpose();
            b_map.noalias() = jacobian_.transpose() * residuals_;
            sum = 2 * residuals_.transpose() * residuals_;
            return sum;
        }

    protected:
        ModelPtr model_;
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