#include <gtest/gtest.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>

#include <duna/stopwatch.hpp>
#include <duna/registration/registration_3dof.h>

#define N_MAP 20
#define N_SOURCE 45
#define N_SOURCE_MERGE 1

using PointT = pcl::PointXYZI;
using PointCloudT = pcl::PointCloud<PointT>;

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

template <typename Scalar>
class SequenceRegistration : public ::testing::Test
{
protected:
    double compareClouds(const PointCloudT::ConstPtr &cloud_a, const PointCloudT::ConstPtr &cloud_b)
    {
        double diff = 0;
        // Find correspondences
        pcl::search::KdTree<PointT> cloud_a_kdtree;
        cloud_a_kdtree.setInputCloud(cloud_a);

        std::vector<int> indices;
        std::vector<float> unused;
        for (const auto &it : cloud_b->points)
        {
            cloud_a_kdtree.nearestKSearch(it, 1, indices, unused);
            diff += (cloud_a->points[indices[0]].getVector3fMap() - it.getVector3fMap()).norm();
        }

        // RMSE
        return diff / (double)cloud_b->size();
    }

public:
    SequenceRegistration()
    {
        std::string map_filename = "/map";
        std::string scan_filename = "/scan";
        PointCloudT::Ptr temp(new PointCloudT);

        target_.reset(new PointCloudT);

        for (int i = 1; i < N_MAP; ++i)
        {
            std::cerr << "Map Loading: " << map_filename + std::to_string(i) << std::endl;
            if (pcl::io::loadPCDFile(TEST_DATA_DIR + map_filename + std::to_string(i) + ".pcd", *temp) != 0)
            {
                std::cerr << "Unable to load dataset\n";
                exit(-1);
            }

            *target_ = *target_ + *temp;
        }

        // Remove points clode to origin
        pcl::PassThrough<PointT> passthrough;
        passthrough.setInputCloud(target_);
        passthrough.setFilterLimits(.1, 100);
        passthrough.setFilterFieldName("x");
        passthrough.filter(*target_);

        PointCloudT::Ptr source_merge(new PointCloudT);

        for (int i = 1; i < N_SOURCE; ++i)
        {
            PointCloudT::Ptr temp_scan(new PointCloudT);
            std::cerr << "Scan Loading: " << scan_filename + std::to_string(i) << std::endl;
            if (pcl::io::loadPCDFile(TEST_DATA_DIR + scan_filename + std::to_string(i) + ".pcd", *temp_scan) != 0)
            {
                std::cerr << "Unable to load dataset\n";
                exit(-1);
            }

            *source_merge += *temp_scan;

            if (i % N_SOURCE_MERGE == 0)
            {
                source_vector_.push_back(source_merge);
                source_merge.reset(new PointCloudT);
            }
        }
        target_kdtree_.reset(new pcl::search::KdTree<PointT>);
    }

protected:
    PointCloudT::Ptr target_;
    std::vector<PointCloudT::Ptr> source_vector_;
    pcl::search::KdTree<PointT>::Ptr target_kdtree_;
};

using ScalarTypes = ::testing::Types<double>;
TYPED_TEST_SUITE(SequenceRegistration, ScalarTypes);

TYPED_TEST(SequenceRegistration, Indoor)
{

    std::cout << "Map size: " << this->target_->size() << std::endl;
    std::cout << "#Scans: " << this->source_vector_.size() << std::endl;

    duna::Registration3DOF<PointT, PointT, TypeParam> registration;

    registration.setInputTarget(this->target_);
    registration.setMaximumICPIterations(50);
    registration.setTargetSearchMethod(this->target_kdtree_);
    registration.setPoint2Plane();
    registration.setMaximumCorrespondenceDistance(0.25);
    registration.setMaximumOptimizerIterations(3);
    Eigen::Matrix<TypeParam, 4, 4> transform;
    transform.setIdentity();

    PointCloudT aligned;
    PointCloudT::Ptr subsampled_input(new PointCloudT);

    utilities::Stopwatch timer;
    PointCloudT::Ptr HD_cloud(new PointCloudT);

    // Copy full map cloud
    *HD_cloud = *this->target_;
    double total_reg_time = 0.0;
    for (int i = 0; i < this->source_vector_.size(); ++i)
    {
        std::cout << "Registering " << i << ": " << this->source_vector_[i]->size() << std::endl;

        pcl::VoxelGrid<PointT> voxel;
        voxel.setInputCloud(this->target_);
        voxel.setLeafSize(0.1, 0.1, 0.1);
        voxel.filter(*this->target_);

        timer.tick();
        this->target_kdtree_->setInputCloud(this->target_);
        timer.tock("KDTree recomputation");

        timer.tick();
        voxel.setInputCloud(this->source_vector_[i]);
        voxel.setLeafSize(0.1, 0.1, 0.1);
        voxel.filter(*subsampled_input);
        timer.tock("Voxel grid.");

        std::cout << "Subsampled # points: " << subsampled_input->size() << std::endl;
        registration.setInputSource(subsampled_input);
        timer.tick();
        registration.align(transform);
        std::cout << "Iterations: " << registration.getFinalIterationsNumber() << "/" << registration.getMaximumICPIterations() << std::endl;
        std::cout << "registration exit code: " << registration.getOptimizationStatus() << std::endl;
        total_reg_time += timer.tock("Registration");

        transform = registration.getFinalTransformation();

        // std::cout << transform << std::endl;
        timer.tick();
        pcl::transformPointCloud(*this->source_vector_[i], aligned, transform);

        *this->target_ = *this->target_ + aligned;
        *HD_cloud = *HD_cloud + aligned;
        timer.tock("Accumulation");
    }
    std::cout << "All registration took: " << total_reg_time << std::endl;

    PointCloudT::Ptr reference_map(new PointCloudT);
    pcl::io::loadPCDFile(TEST_DATA_DIR "/0_sequence_map_reference.pcd", *reference_map);

    double diff = this->compareClouds(reference_map, HD_cloud);

    std::cout << "Diff =  " << diff << std::endl;

    EXPECT_NEAR(diff, 0.0, 1e-2);

    // std::cout << "Filtering: " << HD_cloud->size() << std::endl;
    // pcl::RadiusOutlierRemoval<PointT> ror;
    // ror.setInputCloud(HD_cloud);
    // ror.setRadiusSearch(0.2);
    // ror.setMinNeighborsInRadius(500);
    // ror.filter(*HD_cloud);
    // std::cout << "Filtered to: " << HD_cloud->size() << std::endl;

    std::string final_cloud_filename = "sequence_3dof";
    if (std::is_same<TypeParam, float>::value)
        final_cloud_filename += "_float";
    else
        final_cloud_filename += "_double";

    final_cloud_filename += ".pcd";
    std::cout << "Saving final pointcloud to " << final_cloud_filename << std::endl;
    pcl::io::savePCDFileBinary(final_cloud_filename, *HD_cloud);
}