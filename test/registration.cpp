#include <gtest/gtest.h>

#include <duna/registration/registration.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/search/kdtree.h>
using PointT = pcl::PointXYZ;
using PointCloutT = pcl::PointCloud<PointT>;

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

class RegistrationTest : public ::testing::Test
{
public:
    RegistrationTest()
    {
        source.reset(new PointCloutT);
        target.reset(new PointCloutT);
        target_kdtree.reset(new pcl::search::KdTree<PointT>);
        reference_transform = Eigen::Matrix4d::Identity();

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

protected:
    duna::Registration<PointT, PointT, double> registration;
    PointCloutT::Ptr source;
    PointCloutT::Ptr target;
    pcl::search::KdTree<PointT>::Ptr target_kdtree;
    Eigen::Matrix4d reference_transform;
};

TEST_F(RegistrationTest, SimpleCase)
{

    reference_transform(0, 3) = 1;
    reference_transform(1, 3) = 2;
    reference_transform(2, 3) = 3;

    pcl::transformPointCloud(*target, *source, reference_transform);

    registration.align();

    std::cerr << registration.getFinalTransformation() << std::endl;
}