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
    }

protected:
    PointCloudT::Ptr source;
    PointCloudT::Ptr target;
    pcl::search::KdTree<PointT>::Ptr kdtree;
    Eigen::Matrix4f reference_transform;
};

TEST_F(TestRegistration, SimpleCase)
{

    reference_transform.col(3) = Eigen::Vector4f(-0.5, 0.3, 0.2, 1);
    pcl::transformPointCloud(*target, *source, reference_transform);

    EXPECT_EQ(source->size(), 397);

    // Apply registration
    duna::Registration<PointT, PointT> registration;
    registration.setSourceCloud(source);
    registration.setTargetCloud(target);
    registration.setMaximumICPIterations(10);
    registration.setMaximumCorrespondenceDistance(2);
    registration.setTargetSearchMethod(kdtree);

    registration.align();

    Eigen::Matrix4f final_transform = registration.getFinalTransformation();

    for (int i = 0; i < reference_transform.size(); ++i)
    {
        EXPECT_NEAR(final_transform(i), reference_transform.inverse()(i), TOLERANCE);
    }

    std::cerr << final_transform << std::endl;
    std::cerr << reference_transform.inverse() << std::endl;
}