#pragma once

#include "duna_optimizer/levenberg_marquadt.h"

namespace duna_optimizer {
template <class Scalar>
class DUNA_OPTIMIZER_EXPORT LevenbergMarquadtDynamic
    : public LevenbergMarquadt<Scalar, Eigen::Dynamic> {
 public:
  using HessianMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using ParameterVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  LevenbergMarquadtDynamic(int num_parameters) : num_parameters_(num_parameters) {}

  inline void setNumParameter(int n) { num_parameters_ = n; }

  virtual ~LevenbergMarquadtDynamic() = default;

  virtual void init(Scalar* x0) override;

 protected:
  int num_parameters_;

  using LevenbergMarquadt<Scalar, Eigen::Dynamic>::costs_;
  using LevenbergMarquadt<Scalar, Eigen::Dynamic>::maximum_iterations_;
  using LevenbergMarquadt<Scalar, Eigen::Dynamic>::executed_iterations_;
  using LevenbergMarquadt<Scalar, Eigen::Dynamic>::lm_init_lambda_factor_;
  using LevenbergMarquadt<Scalar, Eigen::Dynamic>::lm_lambda_;
  using LevenbergMarquadt<Scalar, Eigen::Dynamic>::lm_max_iterations_;
  using LevenbergMarquadt<Scalar, Eigen::Dynamic>::x0_map_;
  using LevenbergMarquadt<Scalar, Eigen::Dynamic>::hessian_;
  using LevenbergMarquadt<Scalar, Eigen::Dynamic>::hessian_diagonal_;
  using LevenbergMarquadt<Scalar, Eigen::Dynamic>::b_;
  using LevenbergMarquadt<Scalar, Eigen::Dynamic>::xi_;
  using LevenbergMarquadt<Scalar, Eigen::Dynamic>::cost_hessian_;
  using LevenbergMarquadt<Scalar, Eigen::Dynamic>::cost_b_;
};
}  // namespace duna_optimizer