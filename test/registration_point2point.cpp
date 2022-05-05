#include <gtest/gtest.h>
#include <duna/registration/registration.h>

#include <pcl/search/kdtree.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>

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

    reference_transform.col(3) = Eigen::Vector4f(1,2,3,1);
    pcl::transformPointCloud(*target,*source,reference_transform);

    EXPECT_EQ(source->size(), 397);

    // Apply registration
    duna::Registration<PointT,PointT> registration;
    registration.setSourceCloud(source);
    registration.setTargetCloud(target);
    registration.setMaximumICPIterations(10);
    registration.setMaximumCorrespondenceDistance(20);

    registration.align();

    std::cerr << registration.getFinalTransformation();
    
    

    
}