#ifndef COST_FUNCTION_H
#define COST_FUNCTION_H

#include <exception>
#include <Eigen/Dense>
#include "duna/types.h"

namespace duna
{
    /* This class serves as a Base for cost function implementations.
     */

    template <class Scalar = double>
    class CostFunctionBase
    {
    public:
        CostFunctionBase() = default;

        CostFunctionBase(int num_residuals, int num_model_outputs) : m_num_residuals(num_residuals), m_num_outputs(num_model_outputs)
        {
        }

        CostFunctionBase(const CostFunctionBase &) = delete;
        CostFunctionBase &operator=(const CostFunctionBase &) = delete;
        virtual ~CostFunctionBase() = default;

        void setNumResiduals(int num_residuals) { m_num_residuals = num_residuals; }

        // Initialize the model (optional). Runs before optimization loop.
        virtual void init(const Scalar *x) {}
        // Setup internal state of the model. Runs at the beggining of the optimization loop.
        virtual void setup(const Scalar *x) {}

        virtual Scalar computeCost(const Scalar *x) = 0;
        virtual Scalar linearize(const Scalar *x, Scalar *hessian, Scalar *b) = 0;

    protected:
        int m_num_residuals;
        int m_num_outputs;
    };
}

#endif