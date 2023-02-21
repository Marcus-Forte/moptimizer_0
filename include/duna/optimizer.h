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
        using Ptr = std::shared_ptr<Optimizer>;
        using ConstPtr = std::shared_ptr<const Optimizer>;
        using CostFunctionType = CostFunctionBase<Scalar>;

        Optimizer() : m_maximum_iterations(15) {}
        Optimizer(const Optimizer &) = delete;
        Optimizer &operator=(const Optimizer &) = delete;
        virtual ~Optimizer() = default;

        bool isCostSmall(Scalar cost_sum)
        {
            if (std::abs(cost_sum) < 8 * (std::numeric_limits<Scalar>::epsilon()))
                return true;
            return false;
        }

        inline void setMaximumIterations(int max_iterations)
        {
            if (max_iterations < 0)
                throw std::invalid_argument("Optimization::max_iterations cannot be less than 0.");
            m_maximum_iterations = max_iterations;
        }
        inline unsigned int getMaximumIterations() const { return m_maximum_iterations; }
        inline unsigned int getExecutedIterations() const { return m_executed_iterations; }

        inline bool checkCosts()
        {
            if (costs_.size() == 0)
            {
                std::cerr << "No cost function added!\n";
                throw std::runtime_error("No cost function added!");
            }
            return true;
        }

        inline void addCost(CostFunctionType *cost)
        {
            costs_.push_back(cost);
        }

        // Clear costs from list. Optionally delete them from memory.
        inline void clearCosts(bool delete_costs = false)
        {
            if (delete_costs)
            {
                for (int i = 0; i < costs_.size(); ++i)
                    delete costs_[i];
            }

            costs_.clear();
        }
        virtual OptimizationStatus step(Scalar *x0) = 0;
        virtual OptimizationStatus minimize(Scalar *x0) = 0;
        virtual bool isDeltaSmall(Scalar *x0) = 0;

    protected:
        virtual bool hasConverged() = 0;
        std::vector<CostFunctionType *> costs_;
        unsigned int m_maximum_iterations;
        unsigned int m_executed_iterations;
    };
}

#endif