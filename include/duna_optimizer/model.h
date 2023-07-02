#pragma once

#include <duna_optimizer/exception.h>

#include <memory>

namespace duna_optimizer {

/// @brief Interface definitions for user models.
/// @tparam Scalar Scalar type: float or double.
template <typename Scalar>
class IBaseModel {
 public:
  using Ptr = std::shared_ptr<IBaseModel>;
  using ConstPtr = std::shared_ptr<const IBaseModel>;
  IBaseModel() = default;
  virtual ~IBaseModel() = default;

  /// @brief Function that sets uo the parameter vector. Meant for converting `x` into
  /// another format.  (i.e setting up transform 'T' from state vector 'x')
  /// @param x parameter vector.
  virtual void setup(const Scalar *x) = 0;

  /// @brief Update internal states of the model. (i.e registration correspondences)
  /// @param x parameter vector.
  virtual void update(const Scalar *x) = 0;

  /// @brief Model function
  /// @param x parameter vector.
  /// @param f_x computation result ( f(x) )
  /// @param index
  /// @return true if value was computed successfully.
  virtual bool f(const Scalar *x, Scalar *f_x, unsigned int index) = 0;

  /// @brief Computes both jacobian and function at same time. Usually they depend on
  /// commons functions.
  /// @param x parameter vector.
  /// @param f_x computation result ( f(x) )
  /// @param jacobian
  /// @param index
  /// @return true if value was computed successfully.
  virtual bool f_df(const Scalar *x, Scalar *f_x, Scalar *jacobian, unsigned int index) = 0;

  /// @brief Clone the model and returns a copy as a IBaseModel pointer.
  /// @return Cloned model.
  virtual Ptr clone() const = 0;
};

/// @brief Base model class for models without jacobians.
/// @tparam Scalar
/// @tparam ModelT
template <typename Scalar, class ModelT>
class BaseModel : public IBaseModel<Scalar> {
 public:
  BaseModel() = default;
  virtual ~BaseModel() = default;

  virtual void setup(const Scalar *x) override {}

  virtual void update(const Scalar *x) override {}

  virtual bool f(const Scalar *x, Scalar *residual, unsigned int index) override = 0;

  virtual bool f_df(const Scalar *x, Scalar *f_x, Scalar *jacobian, unsigned int index) final {
    throw duna_optimizer::Exception(
        "Non implemented non-jacobian model function `f_df` being used.");
  }

  std::shared_ptr<IBaseModel<Scalar>> clone() const {
    const auto &copy_cast = static_cast<const ModelT *>(this);
    return std::make_shared<ModelT>(*copy_cast);
  }
};

/// @brief Base model class for models with jacobians.
/// @tparam Scalar
/// @tparam ModelT
template <typename Scalar, class ModelT>
class BaseModelJacobian : public IBaseModel<Scalar> {
 public:
  BaseModelJacobian() = default;
  virtual ~BaseModelJacobian() = default;

  virtual void setup(const Scalar *x) override {}

  virtual void update(const Scalar *x) override {}

  virtual bool f(const Scalar *x, Scalar *f_x, unsigned int index) override {
    throw duna_optimizer::Exception("Non implemented jacobian model function `f` being used.");
  }

  virtual bool f_df(const Scalar *x, Scalar *f_x, Scalar *jacobian,
                    unsigned int index) override = 0;

  std::shared_ptr<IBaseModel<Scalar>> clone() const {
    const auto &copy_cast = static_cast<const ModelT *>(this);
    return std::make_shared<ModelT>(*copy_cast);
  }
};

}  // namespace duna_optimizer