#include <gtest/gtest.h>
#include <duna/stopwatch.hpp>
#include <duna/registration/registration_3dof.h>

#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/search/kdtree.h>
// #include <pcl/registration/icp.h>
// #include <pcl/registration/transformation_estimation_point_to_plane.h>
// #include <pcl/registration/transformation_estimation_point_to_plane_lls.h>
#include <pcl/features/normal_3d.h>

using PointT = pcl::PointNormal;
using PointCloutT = pcl::PointCloud<PointT>;

#define TOLERANCE 1e-2

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

template <typename Scalar>
class Registration3DOFPoint2Plane : public ::testing::Test
{
public:
    Registration3DOFPoint2Plane()
    {

        reference_transform.setIdentity();
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

        registration.setMaximumICPIterations(50);
        registration.setInputSource(source);
        registration.setInputTarget(target);
        registration.setTargetSearchMethod(target_kdtree);
        registration.setMaximumCorrespondenceDistance(10);
        registration.setMaximumOptimizerIterations(3);
        registration.setPoint2Plane();

#ifndef NDEBUG
        pcl::console::setVerbosityLevel(pcl::console::L_VERBOSE);
#endif
    }

    ~Registration3DOFPoint2Plane()
    {
    }

protected:
    duna::Registration3DOF<PointT, PointT, Scalar> registration;
    PointCloutT::Ptr source;
    PointCloutT::Ptr target;
    pcl::search::KdTree<PointT>::Ptr target_kdtree;
    Eigen::Matrix<Scalar, 4, 4> reference_transform;
};

using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(Registration3DOFPoint2Plane, ScalarTypes);

TYPED_TEST(Registration3DOFPoint2Plane, EasyRotation)
{

    Eigen::Matrix<TypeParam, 3, 3> rot;
    rot = Eigen::AngleAxis<TypeParam>(0.2, Eigen::Matrix<TypeParam, 3, 1>::UnitX()) *
          Eigen::AngleAxis<TypeParam>(0.3, Eigen::Matrix<TypeParam, 3, 1>::UnitY()) *
          Eigen::AngleAxis<TypeParam>(0.4, Eigen::Matrix<TypeParam, 3, 1>::UnitZ());

    this->reference_transform.topLeftCorner(3, 3) = rot;

    Eigen::Matrix<TypeParam, 4, 4> reference_transform_inverse = this->reference_transform.inverse();

    utilities::Stopwatch timer;
    timer.tick();
    pcl::transformPointCloud(*this->target, *this->source, this->reference_transform);

    try
    {
        this->registration.align();
    }
    catch (std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
    }
    std::string message("Duna ICP: " + std::to_string(this->registration.getFinalIterationsNumber()) + "/" + std::to_string(this->registration.getMaximumICPIterations()));
    timer.tock(message);

    Eigen::Matrix<TypeParam, 4, 4> final_transform = this->registration.getFinalTransformation();

    std::cerr << final_transform << std::endl;
    std::cerr << reference_transform_inverse << std::endl;

    for (int i = 0; i < this->reference_transform.size(); ++i)
        EXPECT_NEAR(final_transform(i), reference_transform_inverse(i), TOLERANCE);
}

TYPED_TEST(Registration3DOFPoint2Plane, ToughRotation)
{

    Eigen::Matrix<TypeParam, 3, 3> rot;
    rot = Eigen::AngleAxis<TypeParam>(1.5, Eigen::Matrix<TypeParam, 3, 1>::UnitX()) *
          Eigen::AngleAxis<TypeParam>(1.5, Eigen::Matrix<TypeParam, 3, 1>::UnitY()) *
          Eigen::AngleAxis<TypeParam>(1.5, Eigen::Matrix<TypeParam, 3, 1>::UnitZ());

    this->reference_transform.topLeftCorner(3, 3) = rot;

    Eigen::Matrix<TypeParam, 4, 4> reference_transform_inverse = this->reference_transform.inverse();

    utilities::Stopwatch timer;
    timer.tick();
    pcl::transformPointCloud(*this->target, *this->source, this->reference_transform);

    try
    {
        this->registration.align();
    }
    catch (std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
    }
    std::string message("Duna ICP: " + std::to_string(this->registration.getFinalIterationsNumber()) + "/" + std::to_string(this->registration.getMaximumICPIterations()));
    timer.tock(message);

    Eigen::Matrix<TypeParam, 4, 4> final_transform = this->registration.getFinalTransformation();

    std::cerr << final_transform << std::endl;
    std::cerr << reference_transform_inverse << std::endl;

    for (int i = 0; i < this->reference_transform.size(); ++i)
        EXPECT_NEAR(final_transform(i), reference_transform_inverse(i), TOLERANCE);
}