#pragma once

#include "duna/levenberg_marquadt.h"
#include "duna/types.h"
#include "duna_exports.h"

namespace duna {
template <class Scalar>
class DUNA_OPTIMIZER_EXPORT LevenbergMarquadtDynamic
    : public LevenbergMarquadt<Scalar, duna::Dynamic> {
 public:
  using typename LevenbergMarquadt<Scalar, duna::Dynamic>::HessianMatrix;
  using typename LevenbergMarquadt<Scalar, duna::Dynamic>::ParameterVector;

  LevenbergMarquadtDynamic(int num_parameters)
      : num_parameters_(num_parameters) {}
  virtual ~LevenbergMarquadtDynamic() = default;

  virtual OptimizationStatus step(Scalar *x0) override;
  virtual OptimizationStatus minimize(Scalar *x0) override;

  inline void setNumParameter(int n) { num_parameters_ = n; }

 protected:
  int num_parameters_;
  bool isDeltaSmall(Scalar *delta) override;

  using LevenbergMarquadt<Scalar, duna::Dynamic>::costs_;
  using LevenbergMarquadt<Scalar, duna::Dynamic>::m_maximum_iterations;
  using LevenbergMarquadt<Scalar, duna::Dynamic>::m_executed_iterations;
  using LevenbergMarquadt<Scalar, duna::Dynamic>::m_lm_init_lambda_factor_;
  using LevenbergMarquadt<Scalar, duna::Dynamic>::m_lm_lambda;
  using LevenbergMarquadt<Scalar, duna::Dynamic>::m_lm_max_iterations;
};
}  // namespace duna