#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <duna/cost_function.h>
#include <Eigen/Dense>

namespace duna
{
    template <class Scalar = double, int N_PARAMETERS = duna::Dynamic, int N_OUTPUTS = duna::Dynamic>
    class Optimizer 
    {
        public:
        using ParameterVector = Eigen::Matrix<Scalar, N_PARAMETERS,1>;
        using HessianMatrix = Eigen::Matrix<Scalar, N_PARAMETERS,N_PARAMETERS>;
        using JacobianMatrix = Eigen::Matrix<Scalar, N_OUTPUTS, N_PARAMETERS>;
        using CostFunctionType = CostFunctionBase<Scalar,N_PARAMETERS,N_OUTPUTS>;

        Optimizer() = default;
        Optimizer(const Optimizer&) = delete;
        void operator=(const Optimizer&) = delete;
        virtual ~Optimizer() = default;

        void setMaximumIterations(int max_iterations)
        {
            m_maximum_iterations = max_iterations;
        }

        void setCost(CostFunctionType* cost) { m_cost = cost; }
        virtual OptimizationStatus step(ParameterVector& x0) = 0;
        virtual OptimizationStatus minimize(ParameterVector& x0) = 0;

        protected:
        virtual bool hasConverged() = 0;
        CostFunctionType* m_cost;
        int m_maximum_iterations = 15;

    };
}

#endif