#ifndef CPU_OPTIMIZATOR_HPP
#define CPU_OPTIMIZATOR_HPP

#include "optimizator.h"
#include <iostream>
#include "duna_log.h"

// #define USE_GAUSSNEWTON // Gauss Newton

template <int NPARAM>
class GenericOptimizator : public Optimizator<NPARAM>
{
public:
    using VectorN = typename Optimizator<NPARAM>::VectorN; // Generic Vector
    using Optimizator<NPARAM>::m_cost;                     // Cost Function
    using Optimizator<NPARAM>::max_it;
    // template <typename datatype>
    GenericOptimizator(CostFunction<NPARAM> *cost) : Optimizator<NPARAM>(cost) {
        
    }

    virtual ~GenericOptimizator() = default;

    int minimize(VectorN &x0) override;

protected:
    void preprocess() override
    {
        // DUNA_LOG("Calling CPUOptimizator preprocess()\n");
    }

    int testConvergence(const VectorN &delta) override
    {
        double epsilon = delta.array().abs().maxCoeff();
        DUNA_DEBUG_STREAM("epsilon: " << epsilon);

        if (epsilon < 1e-8)
            return 0; // TODO
        return 1;
    }
};

#endif