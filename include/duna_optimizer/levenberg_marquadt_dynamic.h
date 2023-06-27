#pragma once

#include "duna_exports.h"
#include "duna_optimizer/optimizer.h"

namespace duna_optimizer {
template <class Scalar>
class DUNA_OPTIMIZER_EXPORT LevenbergMarquadtDynamic : public Optimizer<Scalar> {
 public:
  using HessianMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using ParameterVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  LevenbergMarquadtDynamic(int num_parameters) : num_parameters_(num_parameters) {
    lm_max_iterations_ = 8;
  }
  virtual ~LevenbergMarquadtDynamic() = default;

  inline void setLevenbergMarquadtIterations(int max_iterations) {
    lm_max_iterations_ = max_iterations;
  }
  inline unsigned int getLevenbergMarquadtIterations() const { return lm_max_iterations_; }

  void reset() {
    logger::log_debug("[LM] Reset");
    lm_init_lambda_factor_ = 1e-9;
    lm_lambda_ = -1.0;
  }

  virtual OptimizationStatus step(Scalar *x0) override;
  virtual OptimizationStatus minimize(Scalar *x0) override;

  inline void setNumParameter(int n) { num_parameters_ = n; }

 protected:
  int num_parameters_;
  bool isDeltaSmall(Scalar *delta) override;
  bool hasConverged() override { return false; }

  using Optimizer<Scalar>::costs_;
  using Optimizer<Scalar>::maximum_iterations_;
  using Optimizer<Scalar>::executed_iterations_;

  Scalar lm_init_lambda_factor_;
  Scalar lm_lambda_;
  unsigned int lm_max_iterations_;
};
}  // namespace duna_optimizer