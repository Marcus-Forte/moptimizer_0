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
    using MatrixN = typename Optimizator<NPARAM>::MatrixN; // Generic Vector
    using Optimizator<NPARAM>::m_cost;                    
    using Optimizator<NPARAM>::m_max_it;
    // template <typename datatype>
    GenericOptimizator(CostFunction<NPARAM> *cost);

    virtual ~GenericOptimizator() = default;
    virtual opt_status minimize(VectorN &x0) override;

protected:
    int testConvergence(const VectorN &delta) override;
};

#endif