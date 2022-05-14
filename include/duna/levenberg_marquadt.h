#ifndef LEVENBERG_MARQUADT_H
#define LEVENBERG_MARQUADT_H

#include "duna/optimizer.h"

namespace duna
{
    template <class Scalar = double, int N_PARAMETERS = duna::Dynamic, int N_OUTPUTS = duna::Dynamic>
    class LevenbergMarquadt : public Optimizer<Scalar, N_PARAMETERS, N_OUTPUTS>
    {
    public:
        using ParameterVector = typename Optimizer<Scalar, N_PARAMETERS, N_OUTPUTS>::ParameterVector;
        using HessianMatrix = typename Optimizer<Scalar, N_PARAMETERS, N_OUTPUTS>::HessianMatrix;
        using JacobianMatrix = typename Optimizer<Scalar, N_PARAMETERS, N_OUTPUTS>::JacobianMatrix;
        using CostFunctionType = typename Optimizer<Scalar, N_PARAMETERS, N_OUTPUTS>::CostFunctionType;

        LevenbergMarquadt()
        {
            reset();
        }

        void reset()
        {
            m_lm_max_iterations = 10;
            m_lm_init_lambda_factor_ = 1e-9;
            m_lm_lambda = -1.0;
        }

        virtual ~LevenbergMarquadt() = default;

        inline void setLevenbergMarquadtIterations(int max_iterations)
        {
            m_lm_max_iterations = max_iterations;
        }

        OptimizationStatus step(Scalar* x0) override;
        OptimizationStatus minimize(Scalar* x0) override;

    protected:
        // TODO
        bool hasConverged()
        {
            return false;
        }

    private:
        Scalar m_lm_init_lambda_factor_;
        Scalar m_lm_lambda;
        int m_lm_max_iterations;

        using Optimizer<Scalar, N_PARAMETERS, N_OUTPUTS>::m_cost;
        using Optimizer<Scalar, N_PARAMETERS, N_OUTPUTS>::m_maximum_iterations;

        // Delta Convergence
        bool isDeltaSmall(ParameterVector &delta)
        {
            Scalar epsilon = delta.array().abs().maxCoeff();

            if (epsilon < sqrt(std::numeric_limits<Scalar>::epsilon()))
                return true;
            return false;
        }
    };
}

#endif