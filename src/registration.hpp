#pragma once


#include <iostream>
#include "generic_optimizator.h"

template <int NPARAM>
class Registration : public GenericOptimizator<NPARAM>
{
public:
    using Optimizator<NPARAM>::m_cost;


    Registration(CostFunction<NPARAM> *cost) : GenericOptimizator<NPARAM>(cost)
    {
    }

    virtual ~Registration() {}

protected:
    void preprocess() override
    {
        std::cout << "Calling registration preprocess()\n";

        // Find correspondences

        for (int i = 0; i < m_cost->getDataSize(); ++i)
        {
        }

        // Compose Correspondence vector
    }
};