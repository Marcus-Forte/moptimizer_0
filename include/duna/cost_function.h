#ifndef COST_FUNCTION_H
#define COST_FUNCTION_H

#include <exception>
#include <Eigen/Dense>
#include "duna/types.h"
#include "duna/model.h"
#include <duna/logging.h>

namespace duna
{
    /* This class uses CostFunctor to process the total summation cost and linearization
     */

    template <class Scalar = double, int N_PARAMETERS = duna::Dynamic, int N_MODEL_OUTPUTS = duna::Dynamic>
    class CostFunctionBase
    {
    public:
        using ParameterVector = Eigen::Matrix<Scalar, N_PARAMETERS, 1>;
        using ResidualVector = Eigen::Matrix<Scalar, N_MODEL_OUTPUTS, 1>;
        using HessianMatrix = Eigen::Matrix<Scalar, N_PARAMETERS, N_PARAMETERS>;
        using JacobianMatrix = Eigen::Matrix<Scalar, N_MODEL_OUTPUTS, N_PARAMETERS>;

        CostFunctionBase() : m_num_outputs(N_MODEL_OUTPUTS)
        {
            m_num_residuals = 0;
        }

        CostFunctionBase(int num_residuals) : m_num_residuals(num_residuals), m_num_outputs(N_MODEL_OUTPUTS)
        {
        }
        CostFunctionBase(const CostFunctionBase &) = delete;
        CostFunctionBase &operator=(const CostFunctionBase &) = delete;
        virtual ~CostFunctionBase() = default;

        virtual Scalar computeCost(const Scalar *x) = 0;
        virtual Scalar linearize(const ParameterVector &x0, HessianMatrix &hessian, ParameterVector &b) = 0;
        void setNumResiduals(int num_residuals) { m_num_residuals = num_residuals; }

    protected:
        int m_num_residuals;
        int m_num_outputs;
    };

    template <class Scalar = double, int N_PARAMETERS = duna::Dynamic, int N_MODEL_OUTPUTS = duna::Dynamic>
    class CostFunction : public CostFunctionBase<Scalar, N_PARAMETERS, N_MODEL_OUTPUTS>
    {
    public:
        using ParameterVector = typename CostFunctionBase<Scalar, N_PARAMETERS, N_MODEL_OUTPUTS>::ParameterVector;
        using ResidualVector = typename CostFunctionBase<Scalar, N_PARAMETERS, N_MODEL_OUTPUTS>::ResidualVector;
        using HessianMatrix = typename CostFunctionBase<Scalar, N_PARAMETERS, N_MODEL_OUTPUTS>::HessianMatrix;
        using JacobianMatrix = typename CostFunctionBase<Scalar, N_PARAMETERS, N_MODEL_OUTPUTS>::JacobianMatrix;

        // TODO change pointer to smartpointer
        CostFunction(Model<Scalar> *model, int num_residuals) : m_model(model),
                                                                residuals(nullptr),
                                                                residuals_plus(nullptr),
                                                                CostFunctionBase<Scalar, N_PARAMETERS, N_MODEL_OUTPUTS>(num_residuals)
        {
            init();
        }

        CostFunction(Model<Scalar> *model) : m_model(model),
                                            residuals(nullptr),
                                            residuals_plus(nullptr)
        {
            init();
        }

        CostFunction(const CostFunction &) = delete;
        CostFunction &operator=(const CostFunction &) = delete;

        ~CostFunction()
        {
            delete[] residuals_data;
            delete[] residuals_plus_data;
        }

        Scalar computeCost(const Scalar *x) override
        {
            Scalar sum = 0;

            m_model->setup(x);
            for (int i = 0; i < m_num_residuals; ++i)
            {
                (*m_model)(x, residuals_data, i);
                sum += residuals.squaredNorm();
            }

            return sum;
        }

        Scalar linearize(const ParameterVector &x0, HessianMatrix &hessian, ParameterVector &b)
        {
            hessian.setZero();
            b.setZero();

            JacobianMatrix jacobian_row;
            Scalar sum = 0.0;

            const Scalar epsilon = 12 * (std::numeric_limits<Scalar>::epsilon());
            // const Scalar epsilon = 0.1;

            for (int i = 0; i < m_num_residuals; ++i)
            {
                m_model->setup(x0.data());
                (*m_model)(x0.data(), residuals_data, i);
                sum += residuals.squaredNorm();

                // TODO preallocate functors for each parameter
                for (int j = 0; j < x0.size(); ++j)
                {
                    ParameterVector x_plus(x0);
                    x_plus[j] += epsilon;

                    m_model->setup(x_plus.data());
                    (*m_model)(x_plus.data(), residuals_plus_data, i);
                    jacobian_row.col(j) = (residuals_plus - residuals) / epsilon;
                }

                if (residuals.hasNaN())
                    throw std::runtime_error("Residual with NaN");

                hessian.template selfadjointView<Eigen::Lower>().rankUpdate(jacobian_row.transpose()); // this sums ? yes
                b += jacobian_row.transpose() * residuals;
            }

            hessian.template triangularView<Eigen::Upper>() = hessian.transpose();

            return sum;
        }

    protected:
        Model<Scalar> *m_model;
        // Holds results for cost computations
        Scalar *residuals_data;
        Scalar *residuals_plus_data;
        Eigen::Map<const ResidualVector> residuals;
        Eigen::Map<const ResidualVector> residuals_plus;

        using CostFunctionBase<Scalar, N_PARAMETERS, N_MODEL_OUTPUTS>::m_num_outputs;
        using CostFunctionBase<Scalar, N_PARAMETERS, N_MODEL_OUTPUTS>::m_num_residuals;

        void init()
        {
            // m_num_outputs = N_MODEL_OUTPUTS;
            residuals_data = new Scalar[m_num_outputs];
            residuals_plus_data = new Scalar[m_num_outputs];

            // Map allocated arrays to eigen types using placement new syntax.
            new (&residuals) Eigen::Map<const ResidualVector>(residuals_data);
            new (&residuals_plus) Eigen::Map<const ResidualVector>(residuals_plus_data);

            if (N_PARAMETERS == -1)
            {
                throw std::runtime_error("Dynamic parameters no yet implemented");
            }
        }
    };
}
#endif