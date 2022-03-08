#ifndef OPTIMIZATOR_HPP
#define OPTIMIZATOR_HPP

#include "cost_function.hpp"

template <int NPARAM>
class Optimizator
{
public:
    enum Status
    {
        MAX_IT_REACHED,
        SMALL_DELTA
    };

    using VectorN = Eigen::Matrix<float, NPARAM, 1>;
    using MatrixN = Eigen::Matrix<float, NPARAM, NPARAM>;

    Optimizator(CostFunction<NPARAM> *cost) : m_cost(cost) {}
    Optimizator(const Optimizator &) = delete;
    virtual ~Optimizator() = default;

    virtual Status minimize(VectorN &x0) = 0;
    inline void setMaxOptimizationIterations(unsigned int max_it) { m_max_it = max_it; }

protected:
    virtual int testConvergence(const VectorN &delta) = 0;

protected:
    CostFunction<NPARAM> *m_cost = 0;
    unsigned int m_max_it = 5;
};

#endif