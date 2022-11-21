#include <gtest/gtest.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>

#include <duna/map/transformation_estimationMAP.h>
#include <duna/stopwatch.hpp>

using PointT = pcl::PointNormal;
using PointCloutT = pcl::PointCloud<PointT>;

using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(RegistrationPoint2Plane3DOFMAP, ScalarTypes);

#define TOLERANCE 1e-2

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

/* This class tests the use of Duna optimizer as a transform estimator */

template <typename Scalar>
class RegistrationPoint2Plane3DOFMAP : public ::testing::Test
{
public:
    RegistrationPoint2Plane3DOFMAP()
    {
        source.reset(new PointCloutT);
        target.reset(new PointCloutT);
        target_kdtree.reset(new pcl::search::KdTree<PointT>);
        reference_transform.setIdentity();

        if (pcl::io::loadPCDFile(TEST_DATA_DIR "/bunny.pcd", *target) != 0)
        {
            throw std::runtime_error("Unable to laod test data 'bunny.pcd'");
        }

        std::cout << "Loaded : " << target->size() << " points\n";

        target_kdtree->setInputCloud(target);

        pcl::NormalEstimation<PointT, PointT> ne;
        ne.setInputCloud(target);
        ne.setSearchMethod(target_kdtree);
        ne.setKSearch(10);
        ne.compute(*target);

        pcl_icp.setInputTarget(target);
        pcl_icp.setMaxCorrespondenceDistance(10);
        pcl_icp.setMaximumIterations(100);
        pcl_icp.setSearchMethodTarget(target_kdtree);

        pcl::console::setVerbosityLevel(pcl::console::L_DEBUG);
    }

protected:
    pcl::IterativeClosestPoint<PointT, PointT, Scalar> pcl_icp;
    typename pcl::registration::TransformationEstimation<PointT, PointT, Scalar>::Ptr estimation;
    PointCloutT::Ptr source;
    PointCloutT::Ptr target;
    pcl::search::KdTree<PointT>::Ptr target_kdtree;
    Eigen::Matrix<Scalar, 4, 4> reference_transform;
};

// PCL fails this one
TYPED_TEST(RegistrationPoint2Plane3DOFMAP, TestMAPWithSetCovariances)
{
    Eigen::Matrix<TypeParam, 3, 3> rot;
    rot = Eigen::AngleAxis<TypeParam>(1.5, Eigen::Matrix<TypeParam, 3, 1>::UnitX()) *
          Eigen::AngleAxis<TypeParam>(1.5, Eigen::Matrix<TypeParam, 3, 1>::UnitY()) *
          Eigen::AngleAxis<TypeParam>(3.4, Eigen::Matrix<TypeParam, 3, 1>::UnitZ());

    this->reference_transform.topLeftCorner(3, 3) = rot;

    Eigen::Matrix<TypeParam, 4, 4> reference_transform_inverse = this->reference_transform.inverse();

    pcl::transformPointCloud(*this->target, *this->source, this->reference_transform);

    // Instantiate estimators
    typename duna::TransformationEstimatorMAP<PointT, PointT, TypeParam>::Ptr duna_transform(new duna::TransformationEstimatorMAP<PointT, PointT, TypeParam>(true));
    Eigen::Matrix<TypeParam, 6, 6> state_covariance;
    srand((unsigned int)time(0));
    state_covariance.setRandom();

    state_covariance = (10) * state_covariance.transpose() * state_covariance; // force symmetry.

    duna_transform->setStateCovariance(state_covariance);
    duna_transform->setMeasurementCovariance(0.01);
    PointCloutT output;

    this->pcl_icp.setInputSource(this->source);
    utilities::Stopwatch timer;

    this->pcl_icp.setTransformationEstimation(duna_transform);

    timer.tick();
    this->pcl_icp.align(output);
    Eigen::Matrix<TypeParam, 4, 4> final_transform_duna = this->pcl_icp.getFinalTransformation();
    timer.tock("DUNA LM");

    // std::cout << "State COV: \n"
    //           << state_covariance << std::endl;
    std::cout << "Updated COV \n"
              << duna_transform->getUpdatedCovariance() << std::endl;

    std::cerr
        << "PCL/DUNA ICP: \n";
    std::cerr << this->pcl_icp.getFinalTransformation() << std::endl;

    for (int i = 0; i < reference_transform_inverse.size(); ++i)
    {
        EXPECT_NEAR(final_transform_duna(i), reference_transform_inverse(i), TOLERANCE);
    }
}

TYPED_TEST(RegistrationPoint2Plane3DOFMAP, TestMAPWithDefaultCovariances)
{
    Eigen::Matrix<TypeParam, 3, 3> rot;
    rot = Eigen::AngleAxis<TypeParam>(1.5, Eigen::Matrix<TypeParam, 3, 1>::UnitX()) *
          Eigen::AngleAxis<TypeParam>(1.5, Eigen::Matrix<TypeParam, 3, 1>::UnitY()) *
          Eigen::AngleAxis<TypeParam>(3.4, Eigen::Matrix<TypeParam, 3, 1>::UnitZ());

    this->reference_transform.topLeftCorner(3, 3) = rot;

    Eigen::Matrix<TypeParam, 4, 4> reference_transform_inverse = this->reference_transform.inverse();

    pcl::transformPointCloud(*this->target, *this->source, this->reference_transform);

    // Instantiate estimators
    typename duna::TransformationEstimatorMAP<PointT, PointT, TypeParam>::Ptr duna_transform(new duna::TransformationEstimatorMAP<PointT, PointT, TypeParam>(true));

    PointCloutT output;

    this->pcl_icp.setInputSource(this->source);
    utilities::Stopwatch timer;

    this->pcl_icp.setTransformationEstimation(duna_transform);

    timer.tick();
    this->pcl_icp.align(output);
    Eigen::Matrix<TypeParam, 4, 4> final_transform_duna = this->pcl_icp.getFinalTransformation();
    timer.tock("DUNA LM");

    // std::cout << "State COV: \n"
    //           << state_covariance << std::endl;
    std::cout << "Updated COV \n"
              << duna_transform->getUpdatedCovariance() << std::endl;

    std::cerr
        << "PCL/DUNA ICP: \n";
    std::cerr << this->pcl_icp.getFinalTransformation() << std::endl;

    for (int i = 0; i < reference_transform_inverse.size(); ++i)
    {
        EXPECT_NEAR(final_transform_duna(i), reference_transform_inverse(i), TOLERANCE);
    }
}
