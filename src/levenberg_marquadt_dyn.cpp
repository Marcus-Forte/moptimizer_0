#include "moptimizer/levenberg_marquadt_dyn.h"

#include "moptimizer/logger.h"

namespace moptimizer {

template <class Scalar>
LevenbergMarquadtDynamic<Scalar>::LevenbergMarquadtDynamic(int num_parameters)
    : num_parameters_(num_parameters), x0_map_(nullptr, num_parameters), lm_max_iterations_(3) {}

template <class Scalar>
LevenbergMarquadtDynamic<Scalar>::~LevenbergMarquadtDynamic() = default;

template <class Scalar>
void LevenbergMarquadtDynamic<Scalar>::prepare(Scalar *x0) {
  lm_init_lambda_factor_ = 1e-9;
  lm_lambda_ = -1.0;

  new (&x0_map_) Eigen::Map<ParametersType>(x0, num_parameters_);
  xi_.resize(num_parameters_);
  b_.resize(num_parameters_);
  hessian_diagonal_.resize(num_parameters_, num_parameters_);
  hessian_.resize(num_parameters_, num_parameters_);
  cost_hessian_.resize(num_parameters_, num_parameters_);
  cost_b_.resize(num_parameters_);
}

template <class Scalar>
OptimizationStatus LevenbergMarquadtDynamic<Scalar>::step(Scalar *x0) {
  return OptimizationStatus::NUMERIC_ERROR;
}

template <class Scalar>
OptimizationStatus LevenbergMarquadtDynamic<Scalar>::minimize(Scalar *x0) {
  this->checkCosts();

  this->prepare(x0);

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

    if (this->isCostSmall(y0)) {
      return OptimizationStatus::CONVERGED;
    }
    hessian_diagonal_ = hessian_.diagonal().asDiagonal();

    if (lm_lambda_ < 0.0)
      lm_lambda_ = lm_init_lambda_factor_ * hessian_.diagonal().array().abs().maxCoeff();

    Scalar nu = 2.0;

    logger_->log(duna::Logger::L_DEBUG,
                 "Internal Iteration --- : it | max | prev_cost | new_cost | "
                 "rho | "
                 "lambda| nu");

    for (int k = 0; k < lm_max_iterations_; ++k) {
      Eigen::LDLT<HessianType> solver(hessian_ + lm_lambda_ * hessian_diagonal_);

      delta_ = solver.solve(-b_);

      // TODO Manifold operation
      xi_ = x0_map_ + delta_;

      Scalar yi = 0;
      for (const auto cost : costs_) yi += cost->computeCost(xi_.data());

      if (std::isnan(yi)) {
        logger_->log(duna::Logger::L_ERROR, "Numeric Error!");
        return OptimizationStatus::NUMERIC_ERROR;
      }

      Scalar rho = (y0 - yi) / delta_.dot(lm_lambda_ * delta_ - b_);
      logger_->log(duna::Logger::L_DEBUG, "Internal Iteration --- : ", k + 1, '/',
                   lm_max_iterations_, ' ', y0, ' ', yi, ' ', rho, ' ', lm_lambda_, ' ', nu);

      if (rho < 0) {
        if (isDeltaSmall(delta_)) {
          logger_->log(duna::Logger::L_DEBUG,
                       "## Small delta reached: ", delta_.array().abs().maxCoeff());
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
template class LevenbergMarquadtDynamic<float>;
template class LevenbergMarquadtDynamic<double>;

}  // namespace moptimizer
