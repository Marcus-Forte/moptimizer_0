#pragma once

#include <tbb/tbb.h>

#include <Eigen/Dense>

#include "duna_optimizer/loss_function/loss_function.h"
#include "duna_optimizer/model.h"

namespace duna_optimizer {
///
template <class Scalar, int model_parameter_dim = Eigen::Dynamic,
          int model_output_dim = Eigen::Dynamic>
class CostComputation {
 private:
  using HessianType = Eigen::Matrix<Scalar, model_parameter_dim, model_parameter_dim>;
  using JacobianType =
      Eigen::Matrix<Scalar, model_output_dim, model_parameter_dim, Eigen::RowMajor>;
  using ResidualType = Eigen::Matrix<Scalar, model_output_dim, 1>;
  using ParameterType = Eigen::Matrix<Scalar, model_parameter_dim, 1>;
  using CovarianceType = Eigen::Matrix<Scalar, model_output_dim, model_output_dim>;

 public:
  using ModelPtr = typename IBaseModel<Scalar>::Ptr;

  // Initialize maps statically
  CostComputation()
      : actual_parameter_dim_(model_parameter_dim), actual_output_dim_(model_output_dim) {}
  explicit CostComputation(int dyn_model_parameter_dim, int dyn_model_output_dim)
      : actual_parameter_dim_(dyn_model_parameter_dim), actual_output_dim_(dyn_model_output_dim) {
    residuals_.resize(actual_output_dim_, 1);
    residuals_plus_.resize(actual_output_dim_, 1);
    jacobian_.resize(actual_output_dim_, actual_parameter_dim_);
  }

  Scalar computeCost(const Scalar *const x, ModelPtr model, int num_elements) {
    model->setup(x);
    Scalar sum = 0.0;

    for (int i = 0; i < num_elements; ++i) {
      if (model->f(x, residuals_.data(), i)) {
        sum += residuals_.transpose() * residuals_;
      }
    }

    return sum;
  }

  Scalar parallelComputeCost(const Scalar *const x, ModelPtr model, int num_elements) {
    model->setup(x);

    return tbb::parallel_reduce(
        tbb::blocked_range<int>(0, num_elements), 0.0f,
        [&](tbb::blocked_range<int> &r, Scalar init) -> Scalar {
          for (auto it = r.begin(); it != r.end(); ++it) {
            if (model->f(x, residuals_.data(), it)) {
              init += residuals_.transpose() * residuals_;
            }
          }
          return init;
        },
        std::plus<Scalar>());
  }

  Scalar computeHessianNumerical(
      const Scalar *const x, const Scalar *const covariance_data,
      const typename duna_optimizer::loss::ILossFunction<Scalar>::Ptr loss_function,
      Scalar *hessian_data, Scalar *b_data, ModelPtr model, int num_elements) {
    // Mappings
    Eigen::Map<const ParameterType> x_map(x, actual_parameter_dim_, 1);
    Eigen::Map<HessianType> hessian_map(hessian_data, actual_parameter_dim_, actual_parameter_dim_);
    Eigen::Map<ParameterType> b_map(b_data, actual_parameter_dim_, 1);
    Eigen::Map<const CovarianceType> covariance_map(covariance_data, actual_output_dim_,
                                                    actual_output_dim_);

    Scalar sum = 0.0;

    const Scalar min_step_size = std::sqrt(std::numeric_limits<Scalar>::epsilon());

    // Step size
    std::vector<Scalar> h(x_map.size());
    std::vector<ParameterType> x_plus(x_map.size(), x_map);
    std::vector<ModelPtr> models_plus(x_map.size());
    for (int j = 0; j < x_map.size(); ++j) {
      h[j] = min_step_size * abs(x_map[j]);

      if (h[j] == 0.0) h[j] = min_step_size;

      x_plus[j][j] += h[j];

      models_plus[j] = model->clone();
      models_plus[j]->setup((x_plus[j]).data());
    }

    model->setup(x);
    // TODO check if at least a few residuals were computed.
    // TODO paralelize.
    hessian_map.setZero();
    b_map.setZero();

    for (int i = 0; i < num_elements; ++i) {
      if (model->f(x, residuals_.data(), i)) {
        for (int j = 0; j < x_map.size(); ++j) {
          models_plus[j]->f(x_plus[j].data(), residuals_plus_.data(), i);
          jacobian_.col(j) = (residuals_plus_ - residuals_) / h[j];
        }

        Scalar w = loss_function->weight(residuals_.squaredNorm());
        // causing double free! hessian_map.template
        // selfadjointView<Eigen::Lower>().rankUpdate(jacobian_.transpose());
        // // H = J^T * J

        hessian_map.noalias() += w * jacobian_.transpose() * covariance_map * jacobian_;
        b_map.noalias() += w * jacobian_.transpose() * covariance_map * residuals_;
        sum += residuals_.transpose() * residuals_;
      }
    }

    // std::cout << "HessianFinal" << hessian_map << std::endl;
    // hessian_map_.template triangularView<Eigen::Upper>() =
    // hessian_map_.transpose();
    // std::cout << "hessian_map_:\n " << hessian_map_ << std::endl;
    return sum;
  }

  Scalar computeHessian(const Scalar *const x, const Scalar *const covariance_data,
                        const typename loss::ILossFunction<Scalar>::Ptr loss_function,
                        Scalar *hessian_data, Scalar *b_data, ModelPtr model, int num_elements) {
    // Mappings
    Eigen::Map<const ParameterType> x_map(x, actual_parameter_dim_, 1);
    Eigen::Map<HessianType> hessian_map(hessian_data, actual_parameter_dim_, actual_parameter_dim_);
    Eigen::Map<ParameterType> b_map(b_data, actual_parameter_dim_, 1);
    Eigen::Map<const CovarianceType> covariance_map(covariance_data, actual_output_dim_,
                                                    actual_output_dim_);
    Scalar sum = 0.0;

    model->setup(x);
    hessian_map.setZero();
    b_map.setZero();

    // TODO check if at least a few residuals were computed.
    // TODO paralelize.
    for (int i = 0; i < num_elements; ++i) {
      if (model->f_df(x, residuals_.data(), jacobian_.data(), i)) {
        Scalar w = loss_function->weight(residuals_.squaredNorm());

        // causing double free! hessian_map.template
        // selfadjointView<Eigen::Lower>().rankUpdate(jacobian_.transpose());
        // // H = J^T * J
        hessian_map.noalias() += w * jacobian_.transpose() * covariance_map * jacobian_;
        b_map.noalias() += w * jacobian_.transpose() * covariance_map * residuals_;
        sum += residuals_.transpose() * residuals_;
      }
    }
    // std::cout << "hessian_map_:\n " << hessian_map_ << std::endl;
    // hessian_map_.template triangularView<Eigen::Upper>() = hessian_map_.transpose();
    return sum;
  }

  // Variables
  ResidualType residuals_;
  ResidualType residuals_plus_;
  JacobianType jacobian_;

  int actual_parameter_dim_;
  int actual_output_dim_;
};
}  // namespace duna_optimizer