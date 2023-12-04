#pragma once

#include "duna_exports.h"
#include "duna_optimizer/delta.h"
#include "duna_optimizer/optimizer.h"

namespace duna_optimizer {
template <class Scalar>
class DUNA_OPTIMIZER_EXPORT LevenbergMarquadtDynamic : public Optimizer<Scalar> {
 public:
  using HessianType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using ParametersType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  explicit LevenbergMarquadtDynamic(int num_parameters);

  virtual ~LevenbergMarquadtDynamic();

  OptimizationStatus step(Scalar *x0) override;
  OptimizationStatus minimize(Scalar *x0) override;
  inline void setLogger(std::shared_ptr<duna::Logger> logger) { logger_ = logger; }
  inline unsigned int getLevenbergMarquadtIterations() const { return lm_max_iterations_; }
  inline void setLevenbergMarquadtIterations(int max_iterations) {
    lm_max_iterations_ = max_iterations;
  }

 protected:
  int num_parameters_;
  bool hasConverged() override { return false; }
  void prepare(Scalar *x) override;

  using Optimizer<Scalar>::costs_;
  using Optimizer<Scalar>::maximum_iterations_;
  using Optimizer<Scalar>::executed_iterations_;
  using Optimizer<Scalar>::logger_;

  Scalar lm_init_lambda_factor_;
  Scalar lm_lambda_;
  unsigned int lm_max_iterations_;

  Eigen::Map<ParametersType> x0_map_;
  HessianType hessian_;
  HessianType hessian_diagonal_;
  ParametersType b_;
  ParametersType xi_;

  HessianType cost_hessian_;
  ParametersType cost_b_;

  ParametersType delta_;
};
}  // namespace duna_optimizer