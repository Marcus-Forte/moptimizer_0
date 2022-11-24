#include <gtest/gtest.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>

#include <duna/map/transformation_estimationMAP.h>
#include <duna/stopwatch.hpp>
#include <duna/registration/scan_matching_3dof.h>

using PointT = pcl::PointNormal;
using PointCloutT = pcl::PointCloud<PointT>;

using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(DunaRegistration, ScalarTypes);

#define TOLERANCE 1e-2

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

template <typename Scalar>
class DunaRegistration : public ::testing::Test
{
public:
    DunaRegistration()
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
    }

protected:
    PointCloutT::Ptr source;
    PointCloutT::Ptr target;
    pcl::search::KdTree<PointT>::Ptr target_kdtree;
    Eigen::Matrix<Scalar, 4, 4> reference_transform;
};

TYPED_TEST(DunaRegistration, TestSimpleRegistration)
{
    Eigen::Matrix<TypeParam, 3, 3> rot;
    rot = Eigen::AngleAxis<TypeParam>(1.5, Eigen::Matrix<TypeParam, 3, 1>::UnitX()) *
          Eigen::AngleAxis<TypeParam>(1.5, Eigen::Matrix<TypeParam, 3, 1>::UnitY()) *
          Eigen::AngleAxis<TypeParam>(3.4, Eigen::Matrix<TypeParam, 3, 1>::UnitZ());

    this->reference_transform.topLeftCorner(3, 3) = rot;

    Eigen::Matrix<TypeParam, 4, 4> reference_transform_inverse = this->reference_transform.inverse();

    pcl::transformPointCloud(*this->target, *this->source, this->reference_transform);

    duna::ScanMatching3DOF<PointT, PointT, TypeParam> matcher;

    Eigen::Matrix<TypeParam, 3, 1> x0;
    x0.setZero();
    // x0[0] = 0.5;
    // x0[1] = 0.5;
    // x0[2] = 0.5;
    duna::logger log;
    matcher.getLogger().setVerbosityLevel(duna::L_DEBUG);
    matcher.setInputSource(this->source);
    matcher.setInputTarget(this->target);
    matcher.setTargetSearchTree(this->target_kdtree);
    matcher.setMaxNumIterations(25);
    matcher.setMaxNumOptIterations(15);
    matcher.setMaxCorrDistance(5);
    matcher.match(x0.data());

    Eigen::Matrix<TypeParam, 4, 4> final_transform = matcher.getFinalTransform();
    // std::cout << "x0 = " << x0 << std::endl;
    // std::cout << matcher.getFinalTransform() << std::endl;
    // std::cout << reference_transform_inverse << std::endl;
    // std::cout << this->reference_transform << std::endl;

    for (int i = 0; i < reference_transform_inverse.size(); ++i)
    {
        EXPECT_NEAR(final_transform(i), reference_transform_inverse(i), TOLERANCE);
    }
}
