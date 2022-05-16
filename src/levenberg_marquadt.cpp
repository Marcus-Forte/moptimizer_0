#include <duna/levenberg_marquadt.h>
#include <duna/logging.h>

#include <iostream>

namespace duna
{

    template <class Scalar, int N_PARAMETERS>
    OptimizationStatus LevenbergMarquadt<Scalar, N_PARAMETERS>::step(Scalar *x0)
    {
        return OptimizationStatus::NUMERIC_ERROR;
    }

    template <class Scalar, int N_PARAMETERS>
    OptimizationStatus LevenbergMarquadt<Scalar, N_PARAMETERS>::minimize(Scalar *x0)
    {
        // std::cout << " Minimizing...\n";

        if (m_cost == 0)
        {
            std::cerr << "no cost object!\n";
            throw std::runtime_error("no cost object.");
        }

        reset();

        Eigen::Map<ParameterVector> x0_map(x0);
        HessianMatrix hessian;
        HessianMatrix hessian_diagonal;
        ParameterVector b;
        ParameterVector xi;

        hessian.setZero();
        b.setZero();

        for (int j = 0; j < m_maximum_iterations; ++j)
        {
            DUNA_DEBUG_STREAM("## Levenberg-Marquadt Iteration: " << j + 1 << "/" << m_maximum_iterations << " ##\n");

            Scalar y0;
            y0 = m_cost->linearize(x0, hessian.data(), b.data());

            if (abs(y0) < std::numeric_limits<Scalar>::epsilon() * 1)
            {
                DUNA_DEBUG("[LM] --- Small Cost reached --- : %e\n", y0);
                return OptimizationStatus::CONVERGED;
            }

            // DUNA_DEBUG_STREAM("[LM] Hessian: " << hessian << std::endl);
            // DUNA_DEBUG_STREAM("[LM] b: " << b << std::endl);

            hessian_diagonal = hessian.diagonal().asDiagonal();
            // hessian_diagonal = HessianMatrix::Identity();

            if (m_lm_lambda < 0.0)
                m_lm_lambda = m_lm_init_lambda_factor_ * hessian.diagonal().array().abs().maxCoeff();

            Scalar nu = 2.0;

            for (int k = 0; k < m_lm_max_iterations; ++k)
            {
                Eigen::LDLT<HessianMatrix> solver(hessian + m_lm_lambda * hessian_diagonal);
                ParameterVector delta = solver.solve(-b);

#ifndef NDEBUG
                fprintf(stderr, "delta: ");
                for (int n = 0; n < x0_map.size(); ++n)
                {
                    fprintf(stderr, "%f ", delta[n]);
                }
                fprintf(stderr, "\n");
#endif

                // DUNA_DEBUG_STREAM("[LM] --- Solver delta: ");
                // DUNA_DEBUG_STREAM(delta << std::endl);

                // if (isDeltaSmall(delta))
                // {
                //     DUNA_DEBUG("[LM] --- Small Delta reached --- : %f\n", delta.norm());
                //     return OptimizationStatus::SMALL_DELTA;
                // }

                // TODO Manifold operation
                xi = x0_map + delta;

                Scalar yi = m_cost->computeCost(xi.data());

                if (std::isnan(yi))
                {
                    DUNA_DEBUG("[LM] --- Numeric Error --- \n");
                    DUNA_DEBUG_STREAM("Hessian: \n"
                                      << hessian << std::endl);
                    DUNA_DEBUG_STREAM("b(residuals): \n"
                                      << b << std::endl);
                    return OptimizationStatus::NUMERIC_ERROR;
                }

                Scalar rho = (y0 - yi) / delta.dot(m_lm_lambda * delta - b);
                DUNA_DEBUG("[LM] Internal Iteration --- : %d/%d | %e %e %f %f %f\n", k + 1, m_lm_max_iterations, y0, yi, rho, m_lm_lambda, nu);

                if (rho < 0)
                {
                    if (isDeltaSmall(delta))
                    {
                        DUNA_DEBUG("[LM] --- Small Delta reached --- : %e\n", delta.norm());
                        return OptimizationStatus::SMALL_DELTA;
                    }

                    m_lm_lambda = nu * m_lm_lambda;
                    nu = 2 * nu;
                    continue;
                }

                x0_map = xi;
                m_lm_lambda = m_lm_lambda * std::max(1.0 / 3.0, 1 - std::pow(2 * rho - 1, 3));
                break;
            }
        }

        return OptimizationStatus::MAXIMUM_ITERATIONS_REACHED;
    }

    template <class Scalar, int N_PARAMETERS>
    bool LevenbergMarquadt<Scalar, N_PARAMETERS>::isDeltaSmall(ParameterVector &delta)
    {
        Scalar epsilon = delta.array().abs().maxCoeff();

        // if (epsilon < sqrt(std::numeric_limits<Scalar>::epsilon()))
        if (epsilon < std::numeric_limits<Scalar>::epsilon() * 100)
            return true;
        return false;
    }

    // Instantiations
    template class LevenbergMarquadt<double, 2>;
    template class LevenbergMarquadt<float, 2>;

    template class LevenbergMarquadt<double, 4>; // powell

    // Registration
    template class LevenbergMarquadt<double, 6>;
    template class LevenbergMarquadt<double, 3>; // 3DOF

    template class LevenbergMarquadt<float, 6>;
    template class LevenbergMarquadt<float, 3>; // 3DOF

} // namespace duna