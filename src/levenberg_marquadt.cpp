#include "duna_optimizer/levenberg_marquadt.h"

#include "duna_optimizer/logger.h"

namespace duna_optimizer {

template <class Scalar, int N_PARAMETERS>
void LevenbergMarquadt<Scalar, N_PARAMETERS>::init(Scalar *x0) {
  logger_->log(duna::Logger::L_DEBUG, "Init");

  new (&x0_map_) Eigen::Map<ParameterVector>(x0, N_PARAMETERS, 1);
  lm_init_lambda_factor_ = 1e-9;
  lm_lambda_ = -1.0;
}

template <class Scalar, int N_PARAMETERS>
OptimizationStatus LevenbergMarquadt<Scalar, N_PARAMETERS>::step(Scalar *x0) {
  return OptimizationStatus::NUMERIC_ERROR;
}

template <class Scalar, int N_PARAMETERS>
OptimizationStatus LevenbergMarquadt<Scalar, N_PARAMETERS>::minimize(Scalar *x0) {
  this->checkCosts();

  this->init(x0);

  for (executed_iterations_ = 0; executed_iterations_ < maximum_iterations_;
       ++executed_iterations_) {
    logger_->log(duna::Logger::L_DEBUG, "Iteration: ", executed_iterations_, '/',
                 maximum_iterations_);

    Scalar y0 = 0;
    hessian_.setZero();
    b_.setZero();

    for (int cost_i = 0; cost_i < costs_.size(); cost_i++) {
      const auto &cost = costs_[cost_i];

      cost_hessian_.setZero();
      cost_b_.setZero();

      cost->update(x0);
      Scalar cost_y = cost->linearize(x0, cost_hessian_.data(), cost_b_.data());
      logger_->log(duna::Logger::L_DEBUG, "Cost(", cost_i, ") = ", cost_y);
      y0 += cost_y;
      hessian_ += cost_hessian_;
      b_ += cost_b_;
    }

    if (this->isCostSmall(y0)) return OptimizationStatus::CONVERGED;
    hessian_diagonal_ = hessian_.diagonal().asDiagonal();

    if (lm_lambda_ < 0.0)
      lm_lambda_ = lm_init_lambda_factor_ * hessian_.diagonal().array().abs().maxCoeff();

    Scalar nu = 2.0;

    logger_->log(duna::Logger::L_DEBUG,
                 "Internal Iteration --- : it | max | prev_cost | new_cost | "
                 "rho | "
                 "lambda| nu");

    for (int k = 0; k < lm_max_iterations_; ++k) {
      Eigen::LDLT<HessianMatrix> solver(hessian_ + lm_lambda_ * hessian_diagonal_);

      ParameterVector delta = solver.solve(-b_);

      // TODO Manifold operation
      xi_ = x0_map_ + delta;

      Scalar yi = 0;
      for (const auto cost : costs_) yi += cost->computeCost(xi_.data());

      if (std::isnan(yi)) {
        logger_->log(duna::Logger::L_ERROR, "Numeric Error!");
        return OptimizationStatus::NUMERIC_ERROR;
      }

      Scalar rho = (y0 - yi) / delta.dot(lm_lambda_ * delta - b_);
      logger_->log(duna::Logger::L_DEBUG, "Internal Iteration --- : ", k + 1, '/',
                   lm_max_iterations_, ' ', y0, ' ', yi, ' ', rho, ' ', lm_lambda_, ' ', nu);

      if (rho < 0) {
        if (isDeltaSmall(delta)) {
          logger_->log(duna::Logger::L_DEBUG,
                       "## Small delta reached: ", delta.array().abs().maxCoeff());
          if (this->isCostSmall(yi))
            return OptimizationStatus::CONVERGED;
          else
            return OptimizationStatus::SMALL_DELTA;
        }

        lm_lambda_ = nu * lm_lambda_;
        nu = 2 * nu;
        continue;
      }

      x0_map_ = xi_;
      lm_lambda_ = lm_lambda_ * std::max(1.0 / 3.0, 1 - std::pow(2 * rho - 1, 3));
      break;
    }
  }

  return OptimizationStatus::MAXIMUM_ITERATIONS_REACHED;
}

// Instantiations

// Dynamic implementations.
template class LevenbergMarquadt<float, -1>;
template class LevenbergMarquadt<double, -1>;

template class LevenbergMarquadt<float, 2>;
template class LevenbergMarquadt<float, 6>;
template class LevenbergMarquadt<float, 3>;  // 3DOF

template class LevenbergMarquadt<double, 4>;  // powell
template class LevenbergMarquadt<double, 2>;
template class LevenbergMarquadt<double, 6>;
template class LevenbergMarquadt<double, 3>;  // 3DOF

}  // namespace duna_optimizer
