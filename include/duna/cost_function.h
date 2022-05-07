#ifndef COST_FUNCTION_H
#define COST_FUNCTION_H

#include <exception>
#include <Eigen/Dense>
#include "duna/types.h"
#include "duna/model.h"
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

        virtual Scalar computeCost(const Scalar *x) = 0;
        virtual Scalar linearize(const ParameterVector &x0, HessianMatrix &hessian, ParameterVector &b, void *dump = nullptr) = 0;
        void setNumResiduals(int num_residuals) { m_num_residuals = num_residuals; }

    protected:
        int m_num_residuals;
        int m_num_outputs;
    };

    // NOTE. We are using Model as a template to be able to call its copy constructors and enable numercial diff.
    template <typename Model, class Scalar = double, int N_PARAMETERS = duna::Dynamic, int N_MODEL_OUTPUTS = duna::Dynamic>
    class CostFunction : public CostFunctionBase<Scalar, N_PARAMETERS, N_MODEL_OUTPUTS>
    {
    public:
        using ParameterVector = typename CostFunctionBase<Scalar, N_PARAMETERS, N_MODEL_OUTPUTS>::ParameterVector;
        using ResidualVector = typename CostFunctionBase<Scalar, N_PARAMETERS, N_MODEL_OUTPUTS>::ResidualVector;
        using HessianMatrix = typename CostFunctionBase<Scalar, N_PARAMETERS, N_MODEL_OUTPUTS>::HessianMatrix;
        using JacobianMatrix = typename CostFunctionBase<Scalar, N_PARAMETERS, N_MODEL_OUTPUTS>::JacobianMatrix;

        // TODO change pointer to smartpointer
        CostFunction(Model *model, int num_residuals) : m_model(model),
                                                        residuals(nullptr),
                                                        residuals_plus(nullptr),
                                                        CostFunctionBase<Scalar, N_PARAMETERS, N_MODEL_OUTPUTS>(num_residuals)
        {
            init();
        }

        CostFunction(Model *model) : m_model(model),
                                     residuals(nullptr),
                                     residuals_plus(nullptr)
        {
            init();
            // TODO remove warning
            std::cout << "Warning, num_residuals not set\n";
        }

        CostFunction(const CostFunction &) = delete;
        CostFunction &operator=(const CostFunction &) = delete;

        ~CostFunction()
        {
            delete[] residuals_data;
            delete[] residuals_plus_data;
        }

        Scalar computeCost(const Scalar *x) override
        {
            Scalar sum = 0;

            m_model->setup(x);
            for (int i = 0; i < m_num_residuals; ++i)
            {
                (*m_model)(x, residuals_data, i);
                sum += residuals.squaredNorm();
            }

            return sum;
        }

        Scalar linearize(const ParameterVector &x0, HessianMatrix &hessian, ParameterVector &b, void *dump) override
        {
            hessian.setZero();
            b.setZero();

            JacobianMatrix jacobian_row;
            Scalar sum = 0.0;

            // const Scalar min_step_size = std::sqrt(std::numeric_limits<Scalar>::epsilon());
            const Scalar min_step_size = 32 * std::numeric_limits<Scalar>::epsilon();
            // const Scalar epsilon = 0.0001;

            // Create a new model for each numerical increment
            std::vector<Model> diff_models(x0.size(), *m_model);
            std::vector<ParameterVector> x_plus(x0.size(), x0);

            // Step size
            Scalar *h = new Scalar[x0.size()];
            const Scalar relDiff = 1e-6;

            //  std::cerr << "epsilon: ";
            for (int j = 0; j < x0.size(); ++j)
            {                
                // const Scalar step_size = relDiff * abs(x0[j]);
                // h[j] = std::max(step_size,min_step_size);

                h[j] = min_step_size; // OVERRIDE
                // std::cerr << h[j] << " ";

                x_plus[j][j] += h[j];
                diff_models[j].setup((x_plus[j]).data());
            }

            // std::cerr << "\n";

            for (int i = 0; i < m_num_residuals; ++i)
            {
                m_model[0].setup(x0.data());
                m_model[0](x0.data(), residuals_data, i);
                sum += residuals.squaredNorm();

                for (int j = 0; j < x0.size(); ++j)
                {
                    diff_models[j](x_plus[j].data(), residuals_plus_data, i);
                    jacobian_row.col(j) = (residuals_plus - residuals) / h[j];
                }

                hessian.template selfadjointView<Eigen::Lower>().rankUpdate(jacobian_row.transpose()); // this sums ? yes
                b += jacobian_row.transpose() * residuals;

                if (dump != nullptr)
                {
                    Eigen::Matrix<Scalar, -1, -1> *jacobian_dump = reinterpret_cast<Eigen::Matrix<Scalar, -1, -1> *>(dump);
                    jacobian_dump->resize(N_MODEL_OUTPUTS * m_num_residuals, x0.size());

                    /// Jacobian block
                    for (int block_col = 0; block_col < jacobian_row.cols(); ++block_col)
                    {
                        for (int block_row = 0; block_row < jacobian_row.rows(); ++block_row)
                        {
                            (*jacobian_dump)(block_row + i * jacobian_row.rows(), block_col) = jacobian_row(block_row, block_col);
                        }
                    }

                    // for (int row = 0; row < jacobian_row.rows(); ++)
                    // jacobian_dump->row(i) = jacobian_row;
                }
            }

            hessian.template triangularView<Eigen::Upper>() = hessian.transpose();

            return sum;
        }

    protected:
        Model *m_model;
        // Holds results for cost computations
        Scalar *residuals_data;
        Scalar *residuals_plus_data;
        Eigen::Map<const ResidualVector> residuals;
        Eigen::Map<const ResidualVector> residuals_plus;

        using CostFunctionBase<Scalar, N_PARAMETERS, N_MODEL_OUTPUTS>::m_num_outputs;
        using CostFunctionBase<Scalar, N_PARAMETERS, N_MODEL_OUTPUTS>::m_num_residuals;

        void init()
        {
            // m_num_outputs = N_MODEL_OUTPUTS;
            residuals_data = new Scalar[m_num_outputs];
            residuals_plus_data = new Scalar[m_num_outputs];

            // Map allocated arrays to eigen types using placement new syntax.
            new (&residuals) Eigen::Map<const ResidualVector>(residuals_data);
            new (&residuals_plus) Eigen::Map<const ResidualVector>(residuals_plus_data);

            if (N_PARAMETERS == -1)
            {
                throw std::runtime_error("Dynamic parameters no yet implemented");
            }
        }
    };
}
#endif