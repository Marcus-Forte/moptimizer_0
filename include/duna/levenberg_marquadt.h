#pragma once

#include "duna/optimizer.h"
#include "duna_exports.h"

namespace duna
{
    template <class Scalar, int N_PARAMETERS>
    class DUNA_OPTIMIZER_EXPORT LevenbergMarquadt : public Optimizer<Scalar>
    {
    public:
        using HessianMatrix = Eigen::Matrix<Scalar, N_PARAMETERS, N_PARAMETERS>;
        using ParameterVector = Eigen::Matrix<Scalar, N_PARAMETERS, 1>;

        LevenbergMarquadt() : m_lm_max_iterations(8)
        {
            reset();
        }

        void reset()
        {
            logger::log_debug("[LM] Reset");
            m_lm_init_lambda_factor_ = 1e-9;
            m_lm_lambda = -1.0;
        }

        virtual ~LevenbergMarquadt() = default;

        inline void setLevenbergMarquadtIterations(int max_iterations) { m_lm_max_iterations = max_iterations; }
        inline unsigned int getLevenbergMarquadtIterations() const { return m_lm_max_iterations; }

        virtual OptimizationStatus step(Scalar *x0) override;
        virtual OptimizationStatus minimize(Scalar *x0) override;

    protected:
        // TODO
        bool hasConverged()
        {
            return false;
        }
        Scalar m_lm_init_lambda_factor_;
        Scalar m_lm_lambda;
        unsigned int m_lm_max_iterations;

        using Optimizer<Scalar>::costs_;
        using Optimizer<Scalar>::m_maximum_iterations;
        using Optimizer<Scalar>::m_executed_iterations;

        // Delta Convergence
        bool isDeltaSmall(Scalar *delta) override;
        // bool isCostSmall(Scalar cost_sum);
    };
}