#include "duna/levenberg_marquadt_dynamic.h"

namespace duna {
template <class Scalar>
bool LevenbergMarquadtDynamic<Scalar>::isDeltaSmall(Scalar *x0) {
  Eigen::Map<ParameterVector> delta(x0, num_parameters_);
  Scalar epsilon = delta.array().abs().maxCoeff();
  if (epsilon < sqrt(std::numeric_limits<Scalar>::epsilon())) return true;
  return false;
}

template <class Scalar>
OptimizationStatus LevenbergMarquadtDynamic<Scalar>::step(Scalar *x0) {
  return OptimizationStatus::NUMERIC_ERROR;
}

template <class Scalar>
OptimizationStatus LevenbergMarquadtDynamic<Scalar>::minimize(Scalar *x0) {
  this->checkCosts();

  this->reset();

  Eigen::Map<ParameterVector> x0_map(x0, num_parameters_);
  HessianMatrix hessian;
  HessianMatrix hessian_diagonal;
  ParameterVector b;
  ParameterVector xi;

  xi.resize(num_parameters_);
  b.resize(num_parameters_);
  hessian_diagonal.resize(num_parameters_, num_parameters_);
  hessian.resize(num_parameters_, num_parameters_);

  for (m_executed_iterations = 0; m_executed_iterations < m_maximum_iterations;
       ++m_executed_iterations) {
    logger::log_debug("[LM] Levenberg-Marquadt Iteration: %d/%d",
                      m_executed_iterations, m_maximum_iterations);

    Scalar y0 = 0;
    hessian.setZero();
    b.setZero();

    for (int cost_i = 0; cost_i < costs_.size(); cost_i++) {
      const auto &cost = costs_[cost_i];
      HessianMatrix cost_hessian;
      ParameterVector cost_b;

      cost_hessian.resize(num_parameters_, num_parameters_);
      cost_b.resize(num_parameters_);

      cost_hessian.setZero();
      cost_b.setZero();

      cost->update(x0);
      logger::log_debug("[LM] Linearize");
      Scalar cost_y = cost->linearize(x0, cost_hessian.data(), cost_b.data());
      logger::log_debug("[LM] Cost(%d) = %e ", cost_i, cost_y);
      y0 += cost_y;
      hessian += cost_hessian;
      b += cost_b;
    }

    if (this->isCostSmall(y0)) return OptimizationStatus::CONVERGED;

    hessian_diagonal = hessian.diagonal().asDiagonal();

    if (m_lm_lambda < 0.0)
      m_lm_lambda = m_lm_init_lambda_factor_ *
                    hessian.diagonal().array().abs().maxCoeff();

    Scalar nu = 2.0;

    logger::log_debug(
        "[LM] Internal Iteration --- : it | max | prev_cost | new_cost | "
        "rho | "
        "lambda| nu");
    for (int k = 0; k < m_lm_max_iterations; ++k) {
      Eigen::LDLT<HessianMatrix> solver(hessian +
                                        m_lm_lambda * hessian_diagonal);

      ParameterVector delta = solver.solve(-b);

      // TODO Manifold operation
      xi = x0_map + delta;

      Scalar yi = 0;

      for (const auto cost : costs_) yi += cost->computeCost(xi.data());

      if (std::isnan(yi)) {
        logger::log_error("[LM] Numeric Error!");
        return OptimizationStatus::NUMERIC_ERROR;
      }

      Scalar rho = (y0 - yi) / delta.dot(m_lm_lambda * delta - b);
      logger::log_debug("[LM] Internal Iteration --- : %d/%d | %e %e %f %f %f",
                        k + 1, m_lm_max_iterations, y0, yi, rho, m_lm_lambda,
                        nu);

      if (rho < 0) {
        if (this->isDeltaSmall(delta.data())) {
          logger::log_debug("## Small delta reached: %e",
                            delta.array().abs().maxCoeff());
          if (this->isCostSmall(yi))
            return OptimizationStatus::CONVERGED;
          else
            return OptimizationStatus::SMALL_DELTA;
        }

        m_lm_lambda = nu * m_lm_lambda;
        nu = 2 * nu;
        continue;
      }

      x0_map = xi;
      m_lm_lambda =
          m_lm_lambda * std::max(1.0 / 3.0, 1 - std::pow(2 * rho - 1, 3));
      break;
    }
  }

  return OptimizationStatus::MAXIMUM_ITERATIONS_REACHED;
}
template class LevenbergMarquadtDynamic<float>;
template class LevenbergMarquadtDynamic<double>;
}  // namespace duna
