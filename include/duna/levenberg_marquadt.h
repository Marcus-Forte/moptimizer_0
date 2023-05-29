#pragma once

#include "duna/levenberg_marquadt_dynamic.h"

namespace duna {
template <class Scalar, int N_PARAMETERS>
class DUNA_OPTIMIZER_EXPORT LevenbergMarquadt
    : public LevenbergMarquadtDynamic<Scalar> {
 public:
  using HessianMatrix = Eigen::Matrix<Scalar, N_PARAMETERS, N_PARAMETERS>;
  using ParameterVector = Eigen::Matrix<Scalar, N_PARAMETERS, 1>;

  LevenbergMarquadt() : LevenbergMarquadtDynamic<Scalar>(-1) {
    LevenbergMarquadtDynamic<Scalar>::reset();
  }

  virtual ~LevenbergMarquadt() = default;

  virtual OptimizationStatus step(Scalar *x0) override;
  virtual OptimizationStatus minimize(Scalar *x0) override;

 protected:
  // TODO

  using LevenbergMarquadtDynamic<Scalar>::costs_;
  using LevenbergMarquadtDynamic<Scalar>::maximum_iterations_;
  using LevenbergMarquadtDynamic<Scalar>::executed_iterations_;
  using LevenbergMarquadtDynamic<Scalar>::lm_init_lambda_factor_;
  using LevenbergMarquadtDynamic<Scalar>::lm_lambda_;
  using LevenbergMarquadtDynamic<Scalar>::lm_max_iterations_;

  // Delta Convergence
  bool isDeltaSmall(Scalar *delta) override;
};
}  // namespace duna