#pragma once

#include <Eigen/Dense>
#include <exception>

#include "duna/covariance/covariance.h"
#include "duna/loss_function/loss_function.h"
#include "duna/model.h"
#include "duna/types.h"

namespace duna {
/* Base class for cost functions. */

template <class Scalar = double>
class CostFunctionBase {
 public:
  using Model = IBaseModel<Scalar>;
  using ModelPtr = typename Model::Ptr;
  using ModelConstPtr = typename Model::ConstPtr;
  using LossFunctionPtr = typename loss::ILossFunction<Scalar>::Ptr;
  using CovariancePtr = typename covariance::ICovariance<Scalar>::Ptr;

  CostFunctionBase() = default;

  CostFunctionBase(ModelPtr model, int num_residuals)
      : model_(model),
        m_num_residuals(num_residuals)

  {
    loss_function_.reset(new loss::NoLoss<Scalar>());
    covariance_.reset(new covariance::IdentityCovariance<Scalar>(1));
  }

  CostFunctionBase(const CostFunctionBase &) = delete;
  CostFunctionBase &operator=(const CostFunctionBase &) = delete;
  virtual ~CostFunctionBase() = default;

  inline void setNumResiduals(int num_residuals) { m_num_residuals = num_residuals; }
  inline void setLossFunction(LossFunctionPtr loss_function) { loss_function_ = loss_function; }

  // Setup internal state of the model. Runs at the beggining of the
  // optimization loop.
  virtual void update(const Scalar *x) { model_->update(x); }

  virtual Scalar computeCost(const Scalar *x) = 0;
  virtual Scalar linearize(const Scalar *x, Scalar *hessian, Scalar *b) = 0;

  // Initialize internal variables.
  virtual void init(const Scalar *x, Scalar *hessian, Scalar *b) = 0;

 protected:
  int m_num_residuals;

  // Interfaces
  ModelPtr model_;
  LossFunctionPtr loss_function_;
  CovariancePtr covariance_;
};
}  // namespace duna