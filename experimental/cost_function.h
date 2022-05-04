#ifndef COST_FUNCTION_H
#define COST_FUNCTION_H

#include "types.h"
#include <exception>
#include <Eigen/Dense>

#include <iostream>

namespace duna
{
    /* This class uses CostFunctor to process the total summation cost and linearization
     */

    template <class Scalar = double, int N_PARAMETERS = duna::Dynamic, int N_MODEL_OUTPUTS = duna::Dynamic>
    class CostFunctionBase
    {
    public:
        using ParameterVector = Eigen::Matrix<Scalar, N_PARAMETERS, 1>;
        using ResidualVector = Eigen::Matrix<Scalar, N_MODEL_OUTPUTS, 1>;
        using HessianMatrix = Eigen::Matrix<Scalar, N_PARAMETERS, N_PARAMETERS>;
        using JacobianMatrix = Eigen::Matrix<Scalar, N_MODEL_OUTPUTS, N_PARAMETERS>;

        CostFunctionBase() = default;
        CostFunctionBase(const CostFunctionBase&) = delete;
        CostFunctionBase& operator=(const CostFunctionBase&) = delete;
        virtual ~CostFunctionBase() = default;

        inline virtual void computeAt(const Scalar *x, Scalar *residuals, int index) = 0;
        virtual Scalar computeCost(const Scalar *x) = 0;
        virtual Scalar linearize(const ParameterVector &x0, HessianMatrix &hessian, ParameterVector &b) = 0;

    protected:
    };

    template <typename CostFunctor, class Scalar = double, int N_PARAMETERS = duna::Dynamic, int N_MODEL_OUTPUTS = duna::Dynamic>
    class CostFunction : public CostFunctionBase<Scalar, N_PARAMETERS, N_MODEL_OUTPUTS>
    {
    public:
        using ParameterVector = typename CostFunctionBase<Scalar, N_PARAMETERS, N_MODEL_OUTPUTS>::ParameterVector;
        using ResidualVector = typename CostFunctionBase<Scalar, N_PARAMETERS, N_MODEL_OUTPUTS>::ResidualVector;
        using HessianMatrix = typename CostFunctionBase<Scalar, N_PARAMETERS, N_MODEL_OUTPUTS>::HessianMatrix;
        using JacobianMatrix = typename CostFunctionBase<Scalar, N_PARAMETERS, N_MODEL_OUTPUTS>::JacobianMatrix;

        CostFunction(CostFunctor *functor, int num_residuals, int num_outputs) : m_functor(functor), m_num_residuals(num_residuals), m_num_outputs(num_outputs)
        {

            residuals_data = new Scalar[num_outputs];
            if (N_PARAMETERS == -1)
            {
                throw std::runtime_error("Dynamic parameters no yet implemented");
                exit(-1);
            }
        }

        CostFunction(const CostFunction&) = delete;
        CostFunction& operator=(const CostFunction&) = delete;

        ~CostFunction(){
            delete[] residuals_data;
        }

        inline void computeAt(const Scalar *x, Scalar *residuals, int index) override
        {
            (*m_functor)(x, residuals, index);
        }

        Scalar computeCost(const Scalar *x) override
        {
            Scalar sum = 0;
            
            Eigen::Map<const Eigen::Matrix<Scalar, N_MODEL_OUTPUTS, 1>> residuals(residuals_data);

            for (int i = 0; i < m_num_residuals; ++i)
            {
                computeAt(x, residuals_data, i);                
                sum += residuals[0] * residuals[0] ;
            }
      
            return sum;
        }

        Scalar linearize(const ParameterVector &x0, HessianMatrix &hessian, ParameterVector &b)
        {
            hessian.setZero();
            b.setZero();

            JacobianMatrix jacobian_row;

            Scalar *residuals_data = new Scalar[m_num_residuals];
            Scalar *residuals_plus_data = new Scalar[m_num_residuals];

            // Map to Eigen            
            Eigen::Map<const ResidualVector> residuals(residuals_data);
            Eigen::Map<const ResidualVector> residuals_plus(residuals_plus_data);

            Scalar sum = 0.0;

            // const Scalar epsilon = 12 * (std::numeric_limits<Scalar>::epsilon());
            const Scalar epsilon = 0.0001;

            for (int i = 0; i < m_num_residuals; ++i)
            {
                computeAt(x0.data(), residuals_data, i);
                sum += residuals_data[0] * residuals_data[0] ;

                for (int j = 0; j < N_PARAMETERS; ++j)
                {
                    ParameterVector x_plus(x0);
                    x_plus[j] += epsilon;

                    computeAt(x_plus.data(), residuals_plus_data, i);

                    jacobian_row.col(j) = (residuals_plus - residuals) / epsilon;
                }

                hessian.template selfadjointView<Eigen::Lower>().rankUpdate(jacobian_row.transpose()); // this sums ? yes
                b += jacobian_row.transpose() * residuals;
            }
            
            hessian.template triangularView<Eigen::Upper>() = hessian.transpose();
            return sum;
        }

    protected:
        CostFunctor *m_functor;
         // Holds results for cost computations
        Scalar *residuals_data;
        const int m_num_residuals;
        const int m_num_outputs;
    };
}
#endif