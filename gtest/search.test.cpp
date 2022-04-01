#include <pcl/search/octree.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <gtest/gtest.h>

#include "duna/registration.h"
using namespace duna;
using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;





int main(int argc, char **argv)
{

    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

class SearchTest : public testing::Test
{
public:
    SearchTest()
    {
        source.reset(new PointCloudT);
        target.reset(new PointCloudT);
        if (pcl::io::loadPCDFile(TEST_DATA_DIR "/bunny.pcd", *target) != 0)
        {
            std::cerr << "Make sure you run the rest at the binaries folder.\n";
        }
        reference_transform.setIdentity();
    }

    virtual ~SearchTest() {}

protected:
    PointCloudT::Ptr source;
    PointCloudT::Ptr target;
    Eigen::Matrix4f reference_transform;
};

TEST_F(SearchTest, UsingOctreeBase)
{

    reference_transform.col(3) = Eigen::Vector4f(-0.5, 0.3, 0.2, 1);    
    pcl::transformPointCloud(*target, *source, reference_transform);



    
    pcl::search::Octree<PointT>::Ptr octree_search(new pcl::search::Octree<PointT>(0.1));
    octree_search->tree_->setInputCloud(target);
    octree_search->tree_->addPointsFromInputCloud();
    octree_search->tree_->deleteTree();
    octree_search->tree_->addPointsFromInputCloud();

    pcl::search::Search<PointT>::Ptr search_method(octree_search);
    
    Registration<6, PointT, PointT>::DatasetType data;
    data.source = source;
    data.target = target;
    data.tgt_search_method = search_method;

    RegistrationCost<6, PointT, PointT> *cost = new RegistrationCost<6, PointT, PointT>(&data);
    Registration<6, PointT, PointT> *registration = new Registration<6, PointT, PointT>(cost);
    registration->setMaxIcpIterations(50);
    registration->setMaxCorrespondenceDistance(1.0);

    registration->minimize();

    Eigen::Matrix4f final_reg_duna = registration->getFinalTransformation();

    std::cerr << "Reference:\n"
              << reference_transform.inverse() << std::endl;
    std::cerr << "Duna:\n"
              << final_reg_duna << std::endl;

    for (int i = 0; i < 16; i++)
    {
        EXPECT_NEAR(reference_transform.inverse()(i), final_reg_duna(i), 0.01);
    }
}

TEST_F(SearchTest, UsingKdreeBase)
{
    reference_transform.col(3) = Eigen::Vector4f(-0.5, 0.3, 0.2, 1);    
    pcl::transformPointCloud(*target, *source, reference_transform);
    
    pcl::search::KdTree<PointT>::Ptr kdtree_search(new pcl::search::KdTree<PointT>);
    kdtree_search->setInputCloud(target);

    pcl::search::Search<PointT>::Ptr search_method(kdtree_search);

    Registration<6, PointT, PointT>::DatasetType data;
    data.source = source;
    data.target = target;
    data.tgt_search_method = search_method;

    RegistrationCost<6, PointT, PointT> *cost = new RegistrationCost<6, PointT, PointT>(&data);
    Registration<6, PointT, PointT> *registration = new Registration<6, PointT, PointT>(cost);
    registration->setMaxIcpIterations(50);
    registration->setMaxCorrespondenceDistance(1.0);

    registration->minimize();

    Eigen::Matrix4f final_reg_duna = registration->getFinalTransformation();

    std::cerr << "Reference:\n"
              << reference_transform.inverse() << std::endl;
    std::cerr << "Duna:\n"
              << final_reg_duna << std::endl;

    for (int i = 0; i < 16; i++)
    {
        EXPECT_NEAR(reference_transform.inverse()(i), final_reg_duna(i), 0.01);
    }

    
}
