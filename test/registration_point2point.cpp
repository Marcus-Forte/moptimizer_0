#include <gtest/gtest.h>
#include <duna/registration/registration.h>

#include <pcl/search/kdtree.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>

#define TOLERANCE 0.01f

using TESTTYPE = double;
using Matrix3 = Eigen::Matrix<TESTTYPE,3,3>;
using Matrix4 = Eigen::Matrix<TESTTYPE,4,4>;
using Vector4 = Eigen::Matrix<TESTTYPE,4,1>;

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

        reference_transform = Matrix4::Identity();
        // Compute KDTree
        kdtree->setInputCloud(target);

        registration.setSourceCloud(source);
        registration.setTargetCloud(target);
        registration.setMaximumICPIterations(150);
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
    Matrix4 reference_transform;
    Matrix4 result_transform;

    // Main API
    duna::Registration<PointT, PointT,TESTTYPE> registration;
};

TEST_F(TestRegistration, Translation6DOF)
{

    reference_transform.col(3) = Vector4(-0.5, 0.3, 0.2, 1);
    pcl::transformPointCloud(*target, *source, reference_transform);

    registration.align();

    result_transform = registration.getFinalTransformation();
}

TEST_F(TestRegistration, Rotation6DOF)
{
    // Rotation
    Matrix3 rot;
    rot = Eigen::AngleAxis<TESTTYPE>(0.2, Eigen::Matrix<TESTTYPE,3,1>::UnitX()) *
          Eigen::AngleAxis<TESTTYPE>(0.8, Eigen::Matrix<TESTTYPE,3,1>::UnitY()) *
          Eigen::AngleAxis<TESTTYPE>(0.6, Eigen::Matrix<TESTTYPE,3,1>::UnitZ());

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
    Matrix3 rot;
    rot = Eigen::AngleAxis<TESTTYPE>(0.2, Eigen::Matrix<TESTTYPE,3,1>::UnitX()) *
          Eigen::AngleAxis<TESTTYPE>(0., Eigen::Matrix<TESTTYPE,3,1>::UnitY()) *
          Eigen::AngleAxis<TESTTYPE>(0.2, Eigen::Matrix<TESTTYPE,3,1>::UnitZ());

    reference_transform.topLeftCorner(3, 3) = rot;
    reference_transform.col(3) = Vector4(-0.5, -0.2, 0.1, 1);

    pcl::transformPointCloud(*target, *source, reference_transform);

    registration.setMaximumOptimizationIterations(3);
    registration.align();

    result_transform = registration.getFinalTransformation();
}

TEST_F(TestRegistration, Tough6DOF)
{

    Matrix3 rot;
    rot = Eigen::AngleAxis<TESTTYPE>(0.7, Eigen::Matrix<TESTTYPE,3,1>::UnitX()) *
          Eigen::AngleAxis<TESTTYPE>(0.7, Eigen::Matrix<TESTTYPE,3,1>::UnitY()) *
          Eigen::AngleAxis<TESTTYPE>(0.7, Eigen::Matrix<TESTTYPE,3,1>::UnitZ());

    reference_transform.topLeftCorner(3, 3) = rot;
    reference_transform.col(3) = Vector4(-0.9, -0.5, 0.5, 1);

    pcl::transformPointCloud(*target, *source, reference_transform);

    registration.setMaximumOptimizationIterations(3);
    registration.align();

    result_transform = registration.getFinalTransformation();

}
