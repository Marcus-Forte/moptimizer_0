#pragma once

#include <Eigen/Dense>
/* This class defines the interface of parameter objects.
  Parameters are the variables to be minimized in the optimization problem.
*/

namespace duna_optimizer {

class IParameter {
 public:
  virtual void Plus(const IParameter *in, IParameter *out) const = 0;
  virtual void Minus(const IParameter *in, IParameter *out) const = 0;
};

// Real numbers usual sum
template <int Dim, typename Scalar = double>
class Parameter : public IParameter {
 public:
  void Plus(const IParameter *in, IParameter *out) const override {
    const Parameter<Dim, Scalar> *in_ = dynamic_cast<const Parameter<Dim, Scalar> *>(in);
    Parameter<Dim, Scalar> *out_ = dynamic_cast<Parameter<Dim, Scalar> *>(out_);
    out_->values_ = this->values_ + in_->values_;
  }

  void Minus(const IParameter *in, IParameter *out) const override {}

  Eigen::Matrix<Scalar, Dim, 1> values_;
};

void plus(const IParameter *lhs, const IParameter *rhs, IParameter *res) { lhs->Plus(rhs, res); }

}  // namespace duna
