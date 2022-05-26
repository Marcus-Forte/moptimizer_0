#include <gtest/gtest.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>

#include <duna/stopwatch.hpp>
#include <duna/registration/registration_3dof.h>

#define N_MAP 20
#define N_SOURCE 100
#define N_SOURCE_MERGE 1

using PointT = pcl::PointXYZI;
using PointCloudT = pcl::PointCloud<PointT>;

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

class SequenceRegistration : public ::testing::Test
{
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

            pcl::PassThrough<PointT> passthrough;
            passthrough.setInputCloud(temp);
            passthrough.setFilterLimits(.1, 100);
            passthrough.setFilterFieldName("x");
            passthrough.filter(*temp);
            *target_ = *target_ + *temp;
        }

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
    std::vector<PointCloudT::Ptr> source_vector_;
    PointCloudT::Ptr target_;
    pcl::search::KdTree<PointT>::Ptr target_kdtree_;
};

TEST_F(SequenceRegistration, Indoor)
{
    std::cout << "Map size: " << target_->size() << std::endl;
    std::cout << "#Scans: " << source_vector_.size() << std::endl;

    duna::Registration3DOF<PointT, PointT, double> registration;

    registration.setInputTarget(target_);
    registration.setMaximumICPIterations(50);
    registration.setTargetSearchMethod(target_kdtree_);
    registration.setPoint2Plane();
    registration.setMaximumCorrespondenceDistance(0.15);
    registration.setMaximumOptimizerIterations(3);
    Eigen::Matrix4d transform;
    transform.setIdentity();

    PointCloudT aligned;
    PointCloudT::Ptr subsampled_input(new PointCloudT);

    utilities::Stopwatch timer;
    PointCloudT::Ptr HD_cloud(new PointCloudT);

    // Copy full map cloud
    *HD_cloud = *target_;
    double total_reg_time = 0.0;
    for (int i = 0; i < source_vector_.size(); ++i)
    {
        std::cout << "Registering " << i << ": " << source_vector_[i]->size() << std::endl;

        pcl::VoxelGrid<PointT> voxel;
        voxel.setInputCloud(target_);
        voxel.setLeafSize(0.1, 0.1, 0.1);
        voxel.filter(*target_);

        timer.tick();
        target_kdtree_->setInputCloud(target_);
        timer.tock("KDTree recomputation");

        timer.tick();
        voxel.setInputCloud(source_vector_[i]);
        voxel.setLeafSize(0.2, 0.2, 0.2);
        voxel.filter(*subsampled_input);
        timer.tock("Voxel grid.");
        
        std::cout << "Subsampled # points: " << subsampled_input->size() << std::endl;
        registration.setInputSource(subsampled_input);
        timer.tick();
        registration.align(transform);
        total_reg_time += timer.tock("Registration");

        transform = registration.getFinalTransformation();

        std::cout << transform << std::endl;
        timer.tick();
        pcl::transformPointCloud(*source_vector_[i], aligned, transform);

        *target_ = *target_ + aligned;
        *HD_cloud = *HD_cloud + aligned;
        timer.tock("Accumulation");
    }
    std::cout << "All registration took: " << total_reg_time;
    std::cout << "Saving final pointcloud\n";
    pcl::io::savePCDFileBinary("sequence3dof_final.pcd", *HD_cloud);
}