#ifndef COST_FUNCTION_H
#define COST_FUNCTION_H

#include <exception>
#include <Eigen/Dense>
#include "duna/types.h"

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
            m_num_residuals = 1;
        }

        CostFunctionBase(int num_residuals) : m_num_residuals(num_residuals), m_num_outputs(N_MODEL_OUTPUTS)
        {
        }

        CostFunctionBase(const CostFunctionBase &) = delete;
        CostFunctionBase &operator=(const CostFunctionBase &) = delete;
        virtual ~CostFunctionBase() = default;

        void setNumResiduals(int num_residuals) { m_num_residuals = num_residuals; }
        
        virtual Scalar computeCost(const Scalar *x, bool setup_data = true) = 0;
        virtual Scalar linearize(const ParameterVector &x0, HessianMatrix &hessian, ParameterVector &b) = 0;

    protected:
        
        int m_num_residuals;
        int m_num_outputs;
    };
}

#endif