#pragma once

#include "duna/optimizator.h"
#include <iostream>
#include "duna/duna_log.h"


namespace duna
{

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
        virtual OptimizationStatus minimize(VectorN &x0) override;

    protected:
        int testConvergence(const VectorN &delta) override;
    };

}