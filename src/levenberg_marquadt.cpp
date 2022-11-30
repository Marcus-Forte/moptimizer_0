#include <duna/levenberg_marquadt.h>

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
        if (costs_.size() == 0)
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

        for (m_executed_iterations = 0; m_executed_iterations < m_maximum_iterations; ++m_executed_iterations)
        {
            logger::log_debug("[LM] Levenberg-Marquadt Iteration: %d/%d", m_executed_iterations, m_maximum_iterations);

            Scalar y0 = 0;
            hessian.setZero();
            b.setZero();

            for (int cost_i = 0; cost_i < costs_.size(); cost_i++)
            {
                const auto& cost = costs_[cost_i];
                HessianMatrix cost_hessian = HessianMatrix::Zero();
                ParameterVector cost_b = ParameterVector::Zero();
                cost->update(x0);
                Scalar cost_y = cost->linearize(x0, cost_hessian.data(), cost_b.data());
                logger::log_debug("[LM] Cost(%d) = %e ", cost_i, cost_y);
                y0 += cost_y;
                hessian += cost_hessian;
                b += cost_b;
            }

            // std::cout << "Hessian: " << hessian << std::endl;
            // std::cout << "b: " << b << std::endl;

            if (isCostSmall(y0))
                return OptimizationStatus::CONVERGED;

            hessian_diagonal = hessian.diagonal().asDiagonal();

            if (m_lm_lambda < 0.0)
                m_lm_lambda = m_lm_init_lambda_factor_ * hessian.diagonal().array().abs().maxCoeff();

            Scalar nu = 2.0;

            logger::log_debug("[LM] Internal Iteration --- : it | max | prev_cost | new_cost | rho | lambda| nu");
            for (int k = 0; k < m_lm_max_iterations; ++k)
            {
                Eigen::LDLT<HessianMatrix> solver(hessian + m_lm_lambda * hessian_diagonal);
                ParameterVector delta = solver.solve(-b);

                // TODO Manifold operation
                xi = x0_map + delta;

                // std::cout << xi << std::endl;

                Scalar yi = 0;

                for (const auto cost : costs_)
                    yi += cost->computeCost(xi.data());

                if (std::isnan(yi))
                {
                    logger::log_error("[LM] Numeric Error!");
                    return OptimizationStatus::NUMERIC_ERROR;
                }

                Scalar rho = (y0 - yi) / delta.dot(m_lm_lambda * delta - b);
                logger::log_debug("[LM] Internal Iteration --- : %d/%d | %e %e %f %f %f", k + 1, m_lm_max_iterations, y0, yi, rho, m_lm_lambda, nu);

                if (rho < 0)
                {
                    if (isDeltaSmall(delta))
                    {
                        logger::log_debug("## Small delta reached: %e", delta.array().abs().maxCoeff());
                        if (isCostSmall(yi))
                            return OptimizationStatus::CONVERGED;
                        else
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
        if (epsilon < sqrt(std::numeric_limits<Scalar>::epsilon()))
            return true;
        return false;
    }

    template <class Scalar, int N_PARAMETERS>
    bool LevenbergMarquadt<Scalar, N_PARAMETERS>::isCostSmall(Scalar cost_sum)
    {
        if (std::abs(cost_sum) < 10 * (std::numeric_limits<Scalar>::epsilon()))
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