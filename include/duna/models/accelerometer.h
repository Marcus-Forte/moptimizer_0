#pragma once
#include <duna/model.h>
#include <duna/so3.h>

#include <Eigen/Dense>

namespace duna {

class Accelerometer : public BaseModelJacobian<double, Accelerometer> {
 public:
  Accelerometer(const double *measurements)
      : measurements_(measurements[0], measurements[1], measurements[2]), gravity_(0, 0, 9.81) {
    transform_.setIdentity();
    // measurements_.normalize();
  }

  void setup(const double *x) override {
    Eigen::Map<const Eigen::Vector3d> x_map(x);
    so3::Exp<double>(x_map, transform_);
  }

  virtual bool f(const double *x, double *f_x, unsigned int index) override {
    // Rotate gravity
    Eigen::Vector3d rotated_gravity = transform_ * gravity_;

    // f_x should be the difference between what was measured by ACC and
    // estimated state!
    f_x[0] = measurements_[0] - rotated_gravity[0];
    f_x[1] = measurements_[1] - rotated_gravity[1];
    f_x[2] = measurements_[2] - rotated_gravity[2];

    // std::cout << "m , e1:" << measurements_[0] << ":" <<
    // rotated_gravity[0]
    // << std::endl; std::cout << "m , e2:" << measurements_[1] << ":" <<
    // rotated_gravity[1] << std::endl; std::cout << "m , e3:" <<
    // measurements_[2] << ":" <<  rotated_gravity[2] << std::endl;

    // // std::cout << "f = " << f_x[0] << "," << f_x[1] << "," << f_x[2] <<
    // std::endl;

    return true;
  }

  bool f_df(const double *x, double *f_x, double *jacobian, unsigned int index) override {
    // fill residue
    f(x, f_x, index);

    Eigen::Map<const Eigen::Vector3d> x_map(x);
    Eigen::Map<Eigen::Matrix3d> jacobian_map(jacobian);
    Eigen::Matrix3d skew;
    Eigen::Matrix3d l_jac;
    skew << SKEW_SYMMETRIC_FROM(gravity_);

    so3::Exp<double>(x_map, transform_);
    so3::leftJacobian<double>(x_map, l_jac);

    Eigen::Vector3d rotated_gravity = transform_ * gravity_;

    skew << SKEW_SYMMETRIC_FROM(rotated_gravity);

    jacobian_map = -skew * l_jac;

    // jacobian_map = skew;

    std::cout << "An Jacobian:\n" << jacobian_map << std::endl;

    // Fill jacobian (3x3)
    return true;
  }

 private:
  Eigen::Matrix3d transform_;
  Eigen::Vector3d gravity_;
  Eigen::Vector3d measurements_;
};
}  // namespace duna