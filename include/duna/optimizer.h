#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <duna/cost_function.h>
#include <Eigen/Dense>

namespace duna
{
    template <class Scalar = double>
    class Optimizer
    {
    public:
        using CostFunctionType = CostFunctionBase<Scalar>;

        Optimizer() : m_maximum_iterations(15) {}
        Optimizer(const Optimizer &) = delete;
        Optimizer &operator=(const Optimizer &) = delete;
        virtual ~Optimizer() = default;

        inline void setMaximumIterations(int max_iterations)
        {
            if (max_iterations < 0)
                throw std::invalid_argument("Optimization::max_iterations cannot be less than 0.");
            m_maximum_iterations = max_iterations;
        }
        inline unsigned int getMaximumIterations() const { return m_maximum_iterations; }
        inline unsigned int getExecutedIterations() const { return m_executed_iterations; }

        void addCost(CostFunctionType *cost)
        {
            costs_.push_back(cost);
        }
        virtual OptimizationStatus step(Scalar *x0) = 0;
        virtual OptimizationStatus minimize(Scalar *x0) = 0;

    protected:
        virtual bool hasConverged() = 0;
        std::vector<CostFunctionType *> costs_;
        unsigned int m_maximum_iterations;
        unsigned int m_executed_iterations;
    };
}

#endif