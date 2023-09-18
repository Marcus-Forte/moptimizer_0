#pragma once

#include "duna_exports.h"
#include "duna_optimizer/delta.h"
#include "duna_optimizer/logger.h"
#include "duna_optimizer/optimizer.h"

namespace duna_optimizer {
template <class Scalar, int N_PARAMETERS>
class DUNA_OPTIMIZER_EXPORT LevenbergMarquadt : public Optimizer<Scalar> {
 public:
  using HessianMatrix = Eigen::Matrix<Scalar, N_PARAMETERS, N_PARAMETERS>;
  using ParameterVector = Eigen::Matrix<Scalar, N_PARAMETERS, 1>;

  LevenbergMarquadt() : x0_map_(0, 0) {
    lm_max_iterations_ = 8;
    logger_ =
        std::make_shared<duna::Logger>(std::cout, duna::Logger::L_ERROR, "Levenberg-Marquadt");
  }
  virtual ~LevenbergMarquadt() = default;

  inline void setLevenbergMarquadtIterations(int max_iterations) {
    lm_max_iterations_ = max_iterations;
  }
  inline unsigned int getLevenbergMarquadtIterations() const { return lm_max_iterations_; }

  virtual void init(Scalar *x0) override;
  OptimizationStatus step(Scalar *x0) override;
  OptimizationStatus minimize(Scalar *x0) override;

  inline void setLogger(std::shared_ptr<duna::Logger> logger) { logger_ = logger; }

 protected:
  bool hasConverged() override { return false; }

  using Optimizer<Scalar>::costs_;
  using Optimizer<Scalar>::maximum_iterations_;
  using Optimizer<Scalar>::executed_iterations_;

  Eigen::Map<ParameterVector> x0_map_;
  HessianMatrix hessian_;
  HessianMatrix hessian_diagonal_;
  ParameterVector b_;
  ParameterVector xi_;

  HessianMatrix cost_hessian_;
  ParameterVector cost_b_;

  Scalar lm_init_lambda_factor_;
  Scalar lm_lambda_;
  unsigned int lm_max_iterations_;

  std::shared_ptr<duna::Logger> logger_;
};
}  // namespace duna_optimizer