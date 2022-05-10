#ifndef COST_FUNCTION_H
#define COST_FUNCTION_H

#include <exception>
#include <Eigen/Dense>
#include "duna/types.h"
// #include "duna/model.h"
#include <duna/logging.h>
#include <vector>

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

        CostFunctionBase() : m_num_outputs(N_MODEL_OUTPUTS)
        {
            m_num_residuals = 0;
        }

        CostFunctionBase(int num_residuals) : m_num_residuals(num_residuals), m_num_outputs(N_MODEL_OUTPUTS)
        {
        }

        CostFunctionBase(const CostFunctionBase &) = delete;
        CostFunctionBase &operator=(const CostFunctionBase &) = delete;
        virtual ~CostFunctionBase() = default;

        virtual Scalar computeCost(const Scalar *x, bool setup_data = true) = 0;
        virtual Scalar linearize(const ParameterVector &x0, HessianMatrix &hessian, ParameterVector &b) = 0;
        void setNumResiduals(int num_residuals) { m_num_residuals = num_residuals; }

    protected:
        int m_num_residuals;
        int m_num_outputs;
    };

    // NOTE. We are using Model as a template to be able to call its copy constructors and enable numercial diff.
    template <typename Model, class Scalar = double, int N_PARAMETERS = duna::Dynamic, int N_MODEL_OUTPUTS = duna::Dynamic>
    class CostFunctionNumDiff : public CostFunctionBase<Scalar, N_PARAMETERS, N_MODEL_OUTPUTS>
    {
    public:
        using ParameterVector = typename CostFunctionBase<Scalar, N_PARAMETERS, N_MODEL_OUTPUTS>::ParameterVector;
        using ResidualVector = typename CostFunctionBase<Scalar, N_PARAMETERS, N_MODEL_OUTPUTS>::ResidualVector;
        using HessianMatrix = typename CostFunctionBase<Scalar, N_PARAMETERS, N_MODEL_OUTPUTS>::HessianMatrix;
        using JacobianMatrix = typename CostFunctionBase<Scalar, N_PARAMETERS, N_MODEL_OUTPUTS>::JacobianMatrix;

        // TODO change pointer to smartpointer
        CostFunctionNumDiff(Model *model, int num_residuals) : m_model(model),
                                                               residuals(nullptr),
                                                               residuals_plus(nullptr),
                                                               residuals_minus(nullptr),
                                                               CostFunctionBase<Scalar, N_PARAMETERS, N_MODEL_OUTPUTS>(num_residuals)
        {
            init();
        }

        CostFunctionNumDiff(Model *model) : m_model(model),
                                            residuals(nullptr),
                                            residuals_plus(nullptr),
                                            residuals_minus(nullptr)
        {
            init();
            // TODO remove warning
            std::cout << "Warning, num_residuals not set\n";
        }

        CostFunctionNumDiff(const CostFunctionNumDiff &) = delete;
        CostFunctionNumDiff &operator=(const CostFunctionNumDiff &) = delete;

        ~CostFunctionNumDiff()
        {
            delete[] residuals_data;
            delete[] residuals_plus_data;
            delete[] residuals_minus_data;
        }

        Scalar computeCost(const Scalar *x, bool setup_data) override
        {
            Scalar sum = 0;

            // TODO make dirty variables?
            if (setup_data)
                m_model->setup(x);

            for (int i = 0; i < m_num_residuals; ++i)
            {
                (*m_model)(x, residuals_data, i);
                sum += residuals.squaredNorm();
            }

            return sum;
        }

        Scalar linearize(const ParameterVector &x0, HessianMatrix &hessian, ParameterVector &b) override
        {
            hessian.setZero();
            b.setZero();

            JacobianMatrix jacobian_row;
            Scalar sum = 0.0;

            const Scalar min_step_size = std::sqrt(std::numeric_limits<Scalar>::epsilon()) ;

            // Create a new model for each numerical increment
            std::vector<Model> diff_plus(x0.size(), *m_model);
            std::vector<Model> diff_minus(x0.size(), *m_model);

            std::vector<ParameterVector> x_plus(x0.size(), x0);
            std::vector<ParameterVector> x_minus(x0.size(), x0);

            // Step size
            Scalar *h = new Scalar[x0.size()];

           
            for (int j = 0; j < x0.size(); ++j)
            {
                h[j] = min_step_size * abs(x0[j]);

                if (h[j] == 0.0)
                    h[j] = min_step_size;

                // TODO Manifold operation
                
                x_plus[j][j] += h[j];
                x_minus[j][j] -= h[j];
                // h[j] = x_plus[j][j] - x0[j];

                diff_plus[j].setup((x_plus[j]).data());
                diff_minus[j].setup((x_minus[j]).data());
            }

            m_model[0].setup(x0.data()); // this was inside the loop below.. Very bad.

            for (int i = 0; i < m_num_residuals; ++i)
            {

                m_model[0](x0.data(), residuals_data, i);
                sum += residuals.squaredNorm();

                for (int j = 0; j < x0.size(); ++j)
                {
                    diff_plus[j](x_plus[j].data(), residuals_plus_data, i);
                    diff_minus[j](x_minus[j].data(), residuals_minus_data, i);

                    jacobian_row.col(j) = (residuals_plus - residuals) / (h[j]);
                }

                // hessian.template selfadjointView<Eigen::Lower>().rankUpdate(jacobian_row.transpose()); // this sums ? yes
                hessian = hessian + (jacobian_row.transpose() * jacobian_row);
                b += jacobian_row.transpose() * residuals;
            }

            delete h;

            hessian.template triangularView<Eigen::Upper>() = hessian.transpose();

            return sum;
        }

    protected:
        Model *m_model;
        // Holds results for cost computations
        Scalar *residuals_data;
        Scalar *residuals_plus_data;
        Scalar *residuals_minus_data;
        Eigen::Map<const ResidualVector> residuals;
        Eigen::Map<const ResidualVector> residuals_plus;
        Eigen::Map<const ResidualVector> residuals_minus;

        using CostFunctionBase<Scalar, N_PARAMETERS, N_MODEL_OUTPUTS>::m_num_outputs;
        using CostFunctionBase<Scalar, N_PARAMETERS, N_MODEL_OUTPUTS>::m_num_residuals;

        void init()
        {
            // m_num_outputs = N_MODEL_OUTPUTS;
            residuals_data = new Scalar[m_num_outputs];
            residuals_plus_data = new Scalar[m_num_outputs];
            residuals_minus_data = new Scalar[m_num_outputs];

            // Map allocated arrays to eigen types using placement new syntax.
            new (&residuals) Eigen::Map<const ResidualVector>(residuals_data);
            new (&residuals_plus) Eigen::Map<const ResidualVector>(residuals_plus_data);
            new (&residuals_minus) Eigen::Map<const ResidualVector>(residuals_minus_data);

            static_assert(N_PARAMETERS != -1, "Dynamic Cost Function not yet implemented");
        }
    };
}
#endif