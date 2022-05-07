#include <gtest/gtest.h>
#include <duna/registration/registration.h>

#include <pcl/search/kdtree.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>

#define TOLERANCE 0.01f

using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

class TestRegistration : public ::testing::Test
{
public:
    TestRegistration()
    {
        source.reset(new PointCloudT);
        target.reset(new PointCloudT);
        kdtree.reset(new pcl::search::KdTree<PointT>);

        if (pcl::io::loadPCDFile(TEST_DATA_DIR "/bunny.pcd", *target) != 0)
        {
            std::cerr << "Make sure you run the rest at the binaries folder.\n";
            throw std::runtime_error("test data 'bunny.pcd' not found");
        }

        EXPECT_EQ(target->size(), 397);

        reference_transform = Eigen::Matrix4f::Identity();
        // Compute KDTree
        kdtree->setInputCloud(target);

        registration.setSourceCloud(source);
        registration.setTargetCloud(target);
        registration.setMaximumICPIterations(50);
        registration.setMaximumCorrespondenceDistance(2);
        registration.setTargetSearchMethod(kdtree);
    }

    ~TestRegistration()
    {

        // Assert results
        for (int i = 0; i < 16; i++)
        {
            EXPECT_NEAR(result_transform(i), reference_transform.inverse()(i), TOLERANCE);
        }

        std::cerr << "Reference (inverse):\n"
                  << reference_transform.inverse() << std::endl;
        std::cerr << "duna opt:\n"
                  << result_transform << std::endl;
    }

protected:
    PointCloudT::Ptr source;
    PointCloudT::Ptr target;
    pcl::search::KdTree<PointT>::Ptr kdtree;
    Eigen::Matrix4f reference_transform;
    Eigen::Matrix4f result_transform;

    // Main API
    duna::Registration<PointT, PointT> registration;
};

TEST_F(TestRegistration, Translation6DOF)
{

    reference_transform.col(3) = Eigen::Vector4f(-0.5, 0.3, 0.2, 1);
    pcl::transformPointCloud(*target, *source, reference_transform);

    registration.align();

    result_transform = registration.getFinalTransformation();
}

TEST_F(TestRegistration, Rotation6DOF)
{
    // Rotation
    Eigen::Matrix3f rot;
    rot = Eigen::AngleAxisf(0.2, Eigen::Vector3f::UnitX()) *
          Eigen::AngleAxisf(0.8, Eigen::Vector3f::UnitY()) *
          Eigen::AngleAxisf(0.6, Eigen::Vector3f::UnitZ());

    reference_transform.topLeftCorner(3, 3) = rot;

    pcl::transformPointCloud(*target, *source, reference_transform);

    // Prepare dataset
    registration.setMaximumOptimizationIterations(3);
    registration.align();

    result_transform = registration.getFinalTransformation();
}

TEST_F(TestRegistration, RotationPlusTranslation6DOF)
{
    // Rotation
    Eigen::Matrix3f rot;
    rot = Eigen::AngleAxisf(0.2, Eigen::Vector3f::UnitX()) *
          Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitY()) *
          Eigen::AngleAxisf(0.2, Eigen::Vector3f::UnitZ());

    reference_transform.topLeftCorner(3, 3) = rot;
    reference_transform.col(3) = Eigen::Vector4f(-0.5, -0.2, 0.1, 1);

    pcl::transformPointCloud(*target, *source, reference_transform);

    registration.setMaximumOptimizationIterations(3);
    registration.align();

    result_transform = registration.getFinalTransformation();
}

TEST_F(TestRegistration, Tough6DOF)
{

    Eigen::Matrix3f rot;
    rot = Eigen::AngleAxisf(0.7, Eigen::Vector3f::UnitX()) *
          Eigen::AngleAxisf(0.7, Eigen::Vector3f::UnitY()) *
          Eigen::AngleAxisf(0.7, Eigen::Vector3f::UnitZ());

    reference_transform.topLeftCorner(3, 3) = rot;
    reference_transform.col(3) = Eigen::Vector4f(-0.9, -0.5, 0.5, 1);

    pcl::transformPointCloud(*target, *source, reference_transform);

    registration.setMaximumOptimizationIterations(3);
    registration.align();

    result_transform = registration.getFinalTransformation();

}
