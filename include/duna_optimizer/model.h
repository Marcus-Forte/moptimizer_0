#pragma once

#include <duna_optimizer/exception.h>

#include <memory>

namespace duna_optimizer {
/* Interface definitions for user models.*/
/* "W can be set to the inverse of the measurement error covariance matrix, in
 * the unusual case that it is known."" */

template <typename Scalar>
class IBaseModel {
 public:
  using Ptr = std::shared_ptr<IBaseModel>;
  using ConstPtr = std::shared_ptr<const IBaseModel>;
  IBaseModel() = default;
  virtual ~IBaseModel() = default;

  // Setups up data for the model (i.e setting up transform 'T' from state
  // vector 'x')
  virtual void setup(const Scalar *x) = 0;

  // Update internal states of the model. (i.e registration correspondences)
  virtual void update(const Scalar *x) = 0;

  // Function (r_i)
  virtual bool f(const Scalar *x, Scalar *f_x, unsigned int index) = 0;

  // Computes both jacobian and function at same time. Usually they depend on
  // commons functions.
  virtual bool f_df(const Scalar *x, Scalar *f_x, Scalar *jacobian, unsigned int index) = 0;

  // Clone method for copying derived classes.
  virtual Ptr clone() const = 0;
};

/* For non-jacobian defined models. */
template <typename Scalar, class ModelT>
class BaseModel : public IBaseModel<Scalar> {
 public:
  BaseModel() = default;
  virtual ~BaseModel() = default;

  // Setups up data for the model (i.e setting up transform 'T' from state
  // vector 'x')
  virtual void setup(const Scalar *x) override {}

  // Update internal states of the model. (i.e registration correspondences)
  virtual void update(const Scalar *x) override {}

  // Function (r_i). Must return true if result if valid.
  virtual bool f(const Scalar *x, Scalar *residual, unsigned int index) override = 0;

  // No jacobian definition.
  virtual bool f_df(const Scalar *x, Scalar *f_x, Scalar *jacobian, unsigned int index) final {
    throw duna_optimizer::Exception(
        "Non implemented non-jacobian model function `f_df` being used.");
  }

  std::shared_ptr<IBaseModel<Scalar>> clone() const {
    auto copy_cast = static_cast<const ModelT *>(this);
    return std::make_shared<ModelT>(*copy_cast);
  }
};

/* For jacobian defined models. */
template <typename Scalar, class ModelT>
class BaseModelJacobian : public IBaseModel<Scalar> {
 public:
  BaseModelJacobian() = default;
  virtual ~BaseModelJacobian() = default;

  // Setups up data for the model (i.e setting up transform 'T' from state
  // vector 'x')
  virtual void setup(const Scalar *x) override {}

  // Update internal states of the model. (i.e registration correspondences)
  virtual void update(const Scalar *x) override {}

  // Function (r_i). Must return true if result if valid.
  virtual bool f(const Scalar *x, Scalar *f_x, unsigned int index) override {
    throw duna_optimizer::Exception("Non implemented jacobian model function `f` being used.");
  }

  // No jacobian definition.
  virtual bool f_df(const Scalar *x, Scalar *f_x, Scalar *jacobian,
                    unsigned int index) override = 0;

  std::shared_ptr<IBaseModel<Scalar>> clone() const {
    auto copy_cast = static_cast<const ModelT *>(this);
    return std::make_shared<ModelT>(*copy_cast);
  }
};

}  // namespace duna_optimizer