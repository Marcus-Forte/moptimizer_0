#include <gtest/gtest.h>

#include <duna/registration/registration.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/search/kdtree.h>
#include <pcl/registration/icp_nl.h>

using PointT = pcl::PointXYZ;
using PointCloutT = pcl::PointCloud<PointT>;

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

template <typename Scalar>
class RegistrationTest : public ::testing::Test
{
public:
    RegistrationTest()
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

        registration.setMaximumICPIterations(50);
        registration.setInputSource(source);
        registration.setInputTarget(target);
        registration.setTargetSearchMethod(target_kdtree);
        registration.setMaximumCorrespondenceDistance(5);
    }

    ~RegistrationTest()
    {
    }

protected:
    duna::Registration<PointT, PointT, Scalar> registration;
    PointCloutT::Ptr source;
    PointCloutT::Ptr target;
    pcl::search::KdTree<PointT>::Ptr target_kdtree;
    Eigen::Matrix<Scalar, 4, 4> reference_transform;
};

using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(RegistrationTest, ScalarTypes);

TYPED_TEST(RegistrationTest, Translation)
{
    this->reference_transform(0, 3) = 1;
    this->reference_transform(1, 3) = 2;
    this->reference_transform(2, 3) = 3;

    Eigen::Matrix<TypeParam, 4, 4> reference_transform_inverse = this->reference_transform.inverse();

    pcl::transformPointCloud(*this->target, *this->source, this->reference_transform);

    try
    {
        this->registration.align();
    }
    catch (std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
    }

    Eigen::Matrix<TypeParam, 4, 4> final_transform = this->registration.getFinalTransformation();
    pcl::console::setVerbosityLevel(pcl::console::L_DEBUG);

    pcl::IterativeClosestPointNonLinear<PointT, PointT> icp;
    icp.setInputSource(this->source);
    icp.setInputTarget(this->target);
    icp.setMaxCorrespondenceDistance(5);
    icp.setMaximumIterations(50);
    icp.setSearchMethodTarget(this->target_kdtree);
    PointCloutT output;
    icp.align(output);

    std::cerr << "PCL ICP: \n";
    std::cerr << icp.getFinalTransformation() << std::endl;

    std::cerr << final_transform << std::endl;
    std::cerr << reference_transform_inverse << std::endl;

    for (int i = 0; i < this->reference_transform.size(); ++i)
        EXPECT_NEAR(final_transform(i), reference_transform_inverse(i), 1e-3);
}

TYPED_TEST(RegistrationTest, Rotation)
{

    Eigen::Matrix<TypeParam, 3, 3> rot;
    rot = Eigen::AngleAxis<TypeParam>(0.2, Eigen::Matrix<TypeParam, 3, 1>::UnitX()) *
          Eigen::AngleAxis<TypeParam>(0.8, Eigen::Matrix<TypeParam, 3, 1>::UnitY()) *
          Eigen::AngleAxis<TypeParam>(0.6, Eigen::Matrix<TypeParam, 3, 1>::UnitZ());

    this->reference_transform.topLeftCorner(3, 3) = rot;
    Eigen::Matrix<TypeParam, 4, 4> reference_transform_inverse = this->reference_transform.inverse();

    pcl::transformPointCloud(*this->target, *this->source, this->reference_transform);

    try
    {
        this->registration.align();
    }
    catch (std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
    }

    Eigen::Matrix<TypeParam, 4, 4> final_transform = this->registration.getFinalTransformation();

    pcl::IterativeClosestPointNonLinear<PointT, PointT> icp;
    icp.setInputSource(this->source);
    icp.setInputTarget(this->target);
    icp.setMaxCorrespondenceDistance(5);
    icp.setMaximumIterations(50);
    icp.setSearchMethodTarget(this->target_kdtree);
    PointCloutT output;
    icp.align(output);

    std::cerr << "PCL ICP: \n";
    std::cerr << icp.getFinalTransformation() << std::endl;

    std::cerr << final_transform << std::endl;
    std::cerr << reference_transform_inverse << std::endl;

    for (int i = 0; i < this->reference_transform.size(); ++i)
        EXPECT_NEAR(final_transform(i), reference_transform_inverse(i), 1e-3);
}