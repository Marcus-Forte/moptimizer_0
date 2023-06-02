#include <duna/cost_function_numerical.h>
#include <duna/levenberg_marquadt.h>
#include <duna/models/scan_matching.h>
#include <gtest/gtest.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/search/kdtree.h>

#include <duna/stopwatch.hpp>

using PointT = pcl::PointNormal;
using PointCloutT = pcl::PointCloud<PointT>;

using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(RegistrationPoint2Plane3DOF, ScalarTypes);

#define TOLERANCE 1e-2

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

template <typename Scalar>
class RegistrationPoint2Plane3DOF : public ::testing::Test {
 public:
  RegistrationPoint2Plane3DOF() {
    source.reset(new PointCloutT);
    target.reset(new PointCloutT);
    target_kdtree.reset(new pcl::search::KdTree<PointT>);
    reference_transform.setIdentity();

    if (pcl::io::loadPCDFile(TEST_DATA_DIR "/bunny.pcd", *target) != 0) {
      throw std::runtime_error("Unable to load test data 'bunny.pcd'");
    }

    std::cout << "Loaded : " << target->size() << " points\n";

    target_kdtree->setInputCloud(target);

    pcl::NormalEstimation<PointT, PointT> ne;
    ne.setInputCloud(target);
    ne.setSearchMethod(target_kdtree);
    ne.setKSearch(10);
    ne.compute(*target);

    duna::logger::setGlobalVerbosityLevel(duna::L_DEBUG);
  }

 protected:
  PointCloutT::Ptr source;
  PointCloutT::Ptr target;
  pcl::search::KdTree<PointT>::Ptr target_kdtree;
  Eigen::Matrix<Scalar, 4, 4> reference_transform;
  Eigen::Matrix<Scalar, 4, 4> result_transform;
  duna::LevenbergMarquadt<Scalar, 3> optimizer;
};

// PCL fails this one
TYPED_TEST(RegistrationPoint2Plane3DOF, DificultRotation) {
  // Arrange
  Eigen::Matrix<TypeParam, 3, 3> rot;
  rot = Eigen::AngleAxis<TypeParam>(1.5, Eigen::Matrix<TypeParam, 3, 1>::UnitX()) *
        Eigen::AngleAxis<TypeParam>(1.5, Eigen::Matrix<TypeParam, 3, 1>::UnitY()) *
        Eigen::AngleAxis<TypeParam>(3.4, Eigen::Matrix<TypeParam, 3, 1>::UnitZ());

  this->reference_transform.topLeftCorner(3, 3) = rot;
  Eigen::Matrix<TypeParam, 4, 4> reference_transform_inverse = this->reference_transform.inverse();
  pcl::transformPointCloud(*this->target, *this->source, this->reference_transform);
  TypeParam x0[3] = {0};

  // Act
  typename duna::ScanMatching3DOFPoint2Plane<PointT, PointT, TypeParam>::Ptr scan_matcher_model;
  scan_matcher_model.reset(new duna::ScanMatching3DOFPoint2Plane<PointT, PointT, TypeParam>(
      this->source, this->target, this->target_kdtree));
  auto cost =
      new duna::CostFunctionNumerical<TypeParam, 3, 1>(scan_matcher_model, this->source->size());
  this->optimizer.addCost(cost);
  this->optimizer.minimize(x0);
  so3::convert3DOFParameterToMatrix(x0, this->result_transform);

  this->optimizer.clearCosts();

  // Assert
  std::cout << "Final x: \n" << Eigen::Map<Eigen::Matrix<TypeParam, 3, 1>>(x0) << std::endl;
  std::cout << "Final Transform: \n" << this->result_transform << std::endl;
  std::cout << "Reference Transform: \n" << reference_transform_inverse << std::endl;

  for (int i = 0; i < reference_transform_inverse.size(); ++i)
    EXPECT_NEAR(this->result_transform(i), reference_transform_inverse(i), TOLERANCE);
}
