#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <filesystem>
#include <fstream>

#include "duna_optimizer/model.h"
#include "duna_optimizer/so3.h"
#include "duna_optimizer/cost_function_analytical_dynamic.h"

using Scalar = double;
using PointT = Eigen::Vector3d;
using PointCloudT = std::vector<PointT>;
using JacobianType = Eigen::Matrix<Scalar, 3, 6>;

#ifndef TEST_DATA_PATH
#error "TEST_DATA_PATH NOT DEFINED!"
#endif

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

class Point2Point : public duna_optimizer::BaseModelJacobian<Scalar, Point2Point> {
 public:
  Point2Point(const PointCloudT &src, const PointCloudT &tgt) : src_pc_(src), tgt_pc_(tgt) {
    transform_.setIdentity();
    std::cout << "src pts: " << src_pc_.size() << std::endl;
    std::cout << "tgt pts: " << tgt_pc_.size() << std::endl;
  }
  virtual ~Point2Point() = default;

  void setup(const Scalar *x) override {}
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

    // Much faster than norm.
    f_x[0] = error[0];
    f_x[1] = error[1];
    f_x[2] = error[2];
    return true;
  }

  bool f_df(const Scalar *x, Scalar *f_x, Scalar *jacobian, unsigned int index) const override {
    const PointT &src_pt = src_pc_[index];
    const PointT &tgt_pt = tgt_pc_[index];

    Eigen::Matrix<Scalar, 4, 1> src_(static_cast<Scalar>(src_pt.x()),
                                     static_cast<Scalar>(src_pt.y()),
                                     static_cast<Scalar>(src_pt.z()), 1.0);
    Eigen::Matrix<Scalar, 4, 1> tgt_(static_cast<Scalar>(tgt_pt.x()),
                                     static_cast<Scalar>(tgt_pt.y()),
                                     static_cast<Scalar>(tgt_pt.z()), 0.0);

    Eigen::Matrix<Scalar, 4, 1> warped_src_ = transform_ * src_;
    warped_src_[3] = 0;

    Eigen::Matrix<Scalar, 4, 1> error = warped_src_ - tgt_;

    f_x[0] = error[0];
    f_x[1] = error[1];
    f_x[2] = error[2];

    Eigen::Map<JacobianType> jacobian_map(jacobian);
    Eigen::Matrix<Scalar, 3, 3> skew;
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

// Test Point2Point analytical diff
TEST(Point2Point, SimpleDistance) {
  PointCloudT src = txt_cloud_loader(std::filesystem::path(TEST_DATA_PATH) / "fachada.txt");
  std::cout << "Loaded : " << src.size() << " points. " << std::endl;
  PointCloudT tgt;
  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();

  Eigen::Matrix3d rot;
  rot = Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitX()) *
        Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitZ());

  transform.topLeftCorner(3, 3) = rot;
  transform(0, 3) = 10.5;
  transform(1, 3) = 10.2;
  transform(2, 3) = 10.3;

  applyTransform(src, tgt, transform);
  // Build problem;

  double x0[6] = {0};
  Point2Point::Ptr model = std::make_shared<Point2Point>(src, tgt);
  duna_optimizer::CostFunctionAnalyticalDynamic<Scalar> cost(model, 6, 3, 1);

  // auto cost_sum = cost.computeCost(x0);
  // std::cout << "cost: " << cost_sum << std::endl;

  // Point2Point model(src, tgt);
}