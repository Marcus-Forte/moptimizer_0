#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <duna/cost_function.h>
#include <duna/logger.h>
#include <Eigen/Dense>
#include <memory>

namespace duna
{
    template <class Scalar = double>
    class Optimizer
    {
    public:
        using CostFunctionType = CostFunctionBase<Scalar>;
        using Ptr = std::shared_ptr<Optimizer>;
        using ConstPtr = std::shared_ptr<const Optimizer>;

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

        inline void addCost(CostFunctionType *cost)
        {
            costs_.push_back(cost);
        }


        inline void clearCosts()
        {
            costs_.clear();
        }
        virtual OptimizationStatus step(Scalar *x0) = 0;
        virtual OptimizationStatus minimize(Scalar *x0) = 0;

        duna::logger &getLogger()
        {
            return logger_;
        }

    protected:
        virtual bool hasConverged() = 0;
        std::vector<CostFunctionType *> costs_;
        unsigned int m_maximum_iterations;
        unsigned int m_executed_iterations;
        duna::logger logger_;
    };
}

#endif