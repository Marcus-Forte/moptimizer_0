#ifndef OPTIMIZATOR_HPP
#define OPTIMIZATOR_HPP

#include "cost_function.hpp"

template <int NPARAM>
class Optimizator
{
public:
    using VectorN = Eigen::Matrix<float, NPARAM, 1>;

    Optimizator(CostFunction<NPARAM> *cost) : m_cost(cost) {}
    virtual ~Optimizator() = default;

    virtual int minimize(VectorN &x0) = 0;

    inline void setMaxIt(unsigned int max)
    {
        max_it = max;
    }

protected:
    virtual void preprocess() = 0;
    virtual int testConvergence(const VectorN& delta) = 0;

protected:
    CostFunction<NPARAM> *m_cost = 0;
    unsigned int max_it = 5;
};

#endif