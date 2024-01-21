#pragma once

#include <Eigen/Dense>
#include <exception>
#include <iostream>

#include "moptimizer/covariance/covariance.h"
#include "moptimizer/loss_function/loss_function.h"
#include "moptimizer/model.h"
#include "moptimizer/types.h"

namespace moptimizer {
/* Base class for cost functions. */

template <class Scalar = double>
class CostFunctionBase {
 public:
  using Model = IBaseModel<Scalar>;
  using ModelPtr = typename Model::Ptr;
  using ModelConstPtr = typename Model::ConstPtr;
  using LossFunctionPtr = typename loss::ILossFunction<Scalar>::Ptr;

  CostFunctionBase(ModelPtr model, int num_residuals)
      : model_(model),
        num_residuals_(num_residuals)

  {
    loss_function_.reset(new loss::NoLoss<Scalar>());
    covariance_.reset(new Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>);
  }
  CostFunctionBase() = delete;

  CostFunctionBase(const CostFunctionBase &) = delete;
  CostFunctionBase &operator=(const CostFunctionBase &) = delete;
  virtual ~CostFunctionBase() = default;

  inline void setLossFunction(LossFunctionPtr loss_function) { loss_function_ = loss_function; }
  inline void setCovariance(const moptimizer::covariance::MatrixPtr<Scalar> covariance) {
    covariance_ = covariance;
  }

  // Setup internal state of the model. Runs at the beggining of the
  // optimization loop.
  virtual void update(const Scalar *x) { model_->update(x); }

  /// @brief Computes  || ∑_i f_i(x) ||²
  /// @param x
  /// @return
  virtual Scalar computeCost(const Scalar *x) = 0;
  virtual Scalar linearize(const Scalar *x, Scalar *hessian, Scalar *b) = 0;

 protected:
  int num_residuals_;

  // Interfaces
  ModelPtr model_;
  LossFunctionPtr loss_function_;
  moptimizer::covariance::MatrixPtr<Scalar> covariance_;
};
}  // namespace moptimizer