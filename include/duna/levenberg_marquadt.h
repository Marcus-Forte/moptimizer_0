#ifndef LEVENBERG_MARQUADT_H
#define LEVENBERG_MARQUADT_H

#include "duna/optimizer.h"

namespace duna
{
    template <class Scalar = double, int N_PARAMETERS = duna::Dynamic>
    class LevenbergMarquadt : public Optimizer<Scalar>
    {
    public:
    using HessianMatrix = Eigen::Matrix<Scalar,N_PARAMETERS,N_PARAMETERS>;
    using ParameterVector = Eigen::Matrix<Scalar,N_PARAMETERS,1>;

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

        inline unsigned int getLevenbergMarquadtIterations() const 
        {
            return m_lm_max_iterations;
        }

        OptimizationStatus step(Scalar *x0) override;
        OptimizationStatus minimize(Scalar *x0) override;

    protected:
        // TODO
        bool hasConverged()
        {
            return false;
        }

    private:
        Scalar m_lm_init_lambda_factor_;
        Scalar m_lm_lambda;
        unsigned int m_lm_max_iterations;
        

        using Optimizer<Scalar>::costs_;
        using Optimizer<Scalar>::m_maximum_iterations;
        using Optimizer<Scalar>::m_executed_iterations;
        using Optimizer<Scalar>::logger_;

        // Delta Convergence
        bool isDeltaSmall(ParameterVector &delta);


    };
}

#endif