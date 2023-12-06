#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <filesystem>
#include <fstream>

#include "duna_optimizer/cost_function_analytical.h"
#include "duna_optimizer/cost_function_numerical.h"
#include "duna_optimizer/cost_function_analytical_dyn.h"
#include "duna_optimizer/cost_function_numerical_dyn.h"
#include "duna_optimizer/levenberg_marquadt_dyn.h"
#include "duna_optimizer/model.h"
#include "duna_optimizer/so3.h"

using Scalar = double;
using PointT = Eigen::Vector3d;
using PointCloudT = std::vector<PointT>;
using JacobianType = Eigen::Matrix<Scalar, 3, 6>;

#ifndef TEST_DATA_PATH
#error "TEST_DATA_PATH NOT DEFINED!"
#endif

class Point2Point : public duna_optimizer::BaseModelJacobian<Scalar, Point2Point> {
 public:
  Point2Point(const PointCloudT &src, const PointCloudT &tgt) : src_pc_(src), tgt_pc_(tgt) {
    transform_.setIdentity();
  }
  virtual ~Point2Point() = default;

  void setup(const Scalar *x) override { so3::convert6DOFParameterToMatrix(x, transform_); }
  bool f(const Scalar *x, Scalar *f_x, unsigned int index) const override {
    const PointT &src_pt = src_pc_[index];
    const PointT &tgt_pt = tgt_pc_[index];

    Eigen::Vector4d src_(static_cast<Scalar>(src_pt.x()), static_cast<Scalar>(src_pt.y()),
                         static_cast<Scalar>(src_pt.z()), 1.0);
    Eigen::Vector4d tgt_(static_cast<Scalar>(tgt_pt.x()), static_cast<Scalar>(tgt_pt.y()),
                         static_cast<Scalar>(tgt_pt.z()), 0.0);

    Eigen::Vector4d warped_src_ = transform_ * src_;
    warped_src_[3] = 0;

    Eigen::Vector4d error = warped_src_ - tgt_;

    // // Much faster than norm.
    f_x[0] = error[0];
    f_x[1] = error[1];
    f_x[2] = error[2];
    return true;
  }

  bool f_df(const Scalar *x, Scalar *f_x, Scalar *jacobian, unsigned int index) const override {
    const PointT &src_pt = src_pc_[index];
    const PointT &tgt_pt = tgt_pc_[index];

    Eigen::Vector4d src_(static_cast<Scalar>(src_pt.x()), static_cast<Scalar>(src_pt.y()),
                         static_cast<Scalar>(src_pt.z()), 1.0);
    Eigen::Vector4d tgt_(static_cast<Scalar>(tgt_pt.x()), static_cast<Scalar>(tgt_pt.y()),
                         static_cast<Scalar>(tgt_pt.z()), 0.0);

    Eigen::Vector4d warped_src_ = transform_ * src_;
    warped_src_[3] = 0;

    Eigen::Vector4d error = warped_src_ - tgt_;

    f_x[0] = error[0];
    f_x[1] = error[1];
    f_x[2] = error[2];

    Eigen::Map<JacobianType> jacobian_map(jacobian);
    Eigen::Matrix3d skew;
    skew << SKEW_SYMMETRIC_FROM(src_);
    jacobian_map.template block<3, 3>(0, 0) = Eigen::Matrix<Scalar, 3, 3>::Identity();
    jacobian_map.template block<3, 3>(0, 3) = -1.0 * skew;

    return true;
  }

 private:
  Eigen::Matrix4d transform_;
  const PointCloudT &src_pc_;
  const PointCloudT &tgt_pc_;
};

class TestPoint2Point : public ::testing::Test {
 public:
  TestPoint2Point() {
    src = txt_cloud_loader(std::filesystem::path(TEST_DATA_PATH) / "fachada.txt");
    std::cout << "Loaded : " << src.size() << " points. " << std::endl;
    transform = Eigen::Matrix4d::Identity();

    Eigen::Matrix3d rot;
    rot = Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitX()) *
          Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitZ());

    transform.topLeftCorner(3, 3) = rot;
    transform(0, 3) = 10.5;
    transform(1, 3) = 10.2;
    transform(2, 3) = 0.1;

    applyTransform(src, tgt, transform);
  }

 protected:
  Eigen::Matrix4d transform;
  PointCloudT src;
  PointCloudT tgt;

 private:
  void applyTransform(const PointCloudT &input, PointCloudT &output,
                      const Eigen::Matrix4d &transform) {
    output.resize(input.size());
    int idx = 0;
    std::for_each(input.cbegin(), input.cend(), [&](const PointT &pt) {
      Eigen::Vector4d pt_4(pt[0], pt[1], pt[2], 1.0);
      auto tf_pt = transform * pt_4;
      output[idx][0] = tf_pt[0];
      output[idx][1] = tf_pt[1];
      output[idx][2] = tf_pt[2];
      idx++;
    });
  }
  PointCloudT txt_cloud_loader(const std::filesystem::path &file) {
    std::cout << "Loading: " << file;
    if (!std::filesystem::is_regular_file(file)) throw std::runtime_error("not a file! exiting");
    PointCloudT cloud;

    std::ifstream cloud_file(file);
    double x, y, z;
    double discard;
    while (cloud_file >> x >> y >> z >> discard >> discard >> discard) {
      // std::cout << x << " " << y <<  " " << z << std::endl;
      cloud.push_back(PointT(x, y, z));
    }
    return cloud;
  }
};

// Test Point2Point analytical diff
TEST_F(TestPoint2Point, ConsistencyOverCostsClasses) {
  // Build problem;

  double x0[6] = {0};
  Point2Point::Ptr model = std::make_shared<Point2Point>(src, tgt);

  int num_residuals = src.size();  // src.size();
  duna_optimizer::CostFunctionAnalytical<Scalar, 6, 3> cost_an_s(model, num_residuals);
  duna_optimizer::CostFunctionAnalyticalDynamic<Scalar> cost_an_d(model, 6, 3, num_residuals);
  duna_optimizer::CostFunctionNumerical<Scalar, 6, 3> cost_num_s(model, num_residuals);
  duna_optimizer::CostFunctionNumericalDynamic<Scalar> cost_num_d(model, 6, 3, num_residuals);

  Eigen::Matrix<Scalar, 6, 6> hessian_an_s;
  Eigen::Matrix<Scalar, 6, 6> hessian_an_d;
  Eigen::Matrix<Scalar, 6, 6> hessian_num_s;
  Eigen::Matrix<Scalar, 6, 6> hessian_num_d;

  Eigen::Matrix<Scalar, 6, 1> b_an_s;
  Eigen::Matrix<Scalar, 6, 1> b_an_d;
  Eigen::Matrix<Scalar, 6, 1> b_num_s;
  Eigen::Matrix<Scalar, 6, 1> b_num_b;

  auto sum_an_s = cost_an_s.linearize(x0, hessian_an_s.data(), b_an_s.data());
  auto sum_an_d = cost_an_d.linearize(x0, hessian_an_d.data(), b_an_d.data());
  auto sum_num_s = cost_num_s.linearize(x0, hessian_num_s.data(), b_num_s.data());
  auto sum_num_d = cost_num_d.linearize(x0, hessian_num_d.data(), b_num_b.data());

  std::cout << sum_an_s << std::endl;
  std::cout << sum_an_d << std::endl;
  std::cout << sum_num_s << std::endl;
  std::cout << sum_num_d << std::endl;

  EXPECT_NEAR(sum_an_s, sum_an_d, 1e-7);
  EXPECT_NEAR(sum_an_s, sum_num_s, 1e-7);
  EXPECT_NEAR(sum_an_s, sum_num_d, 1e-7);

  for (int i = 0; i < 35; ++i) {
    EXPECT_NEAR(hessian_an_s(i), hessian_an_d(i), 1e-7);
  }

  for (int i = 0; i < 35; ++i) {
    EXPECT_NEAR(hessian_num_s(i), hessian_num_d(i), 1e-7);
  }

  // for (int i = 0; i < 35; ++i) {
  //   EXPECT_NEAR(hessian_an_s(i), hessian_num_d(i), 1e-7);
  // }
}

// Test Point2Point analytical diff
TEST_F(TestPoint2Point, Optimization) {
  std::cout << "Transform: " << transform << std::endl;
  Eigen::Matrix<Scalar, 6, 1> x0;
  x0.setZero();
  Point2Point::Ptr model = std::make_shared<Point2Point>(src, tgt);

  int num_residuals = src.size();  // src.size();

  duna_optimizer::CostFunctionNumericalDynamic<Scalar> cost_num_d(model, 6, 3, num_residuals);

  duna_optimizer::LevenbergMarquadtDynamic<Scalar> lm_d(6);
  lm_d.setMaximumIterations(50);

  lm_d.addCost(&cost_num_d);
  lm_d.minimize(x0.data());
  lm_d.clearCosts();
  std::cout << x0;
  x0.setZero();

  duna_optimizer::CostFunctionNumerical<Scalar, 6, 3> cost_num_s(model, num_residuals);
  lm_d.addCost(&cost_num_s);
  lm_d.minimize(x0.data());
  lm_d.clearCosts();
  std::cout << x0;
  x0.setZero();
}