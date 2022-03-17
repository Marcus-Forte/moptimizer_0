#include <gtest/gtest.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/octree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/distances.h>

#include <duna/registration.h>

#include <chrono>

using namespace duna;
using PointT = pcl::PointXYZINormal;
using PointCloudT = pcl::PointCloud<PointT>;

#define N_SCANS_MAP 20
#define N_SCANS_LASER 50

// Accumulate map scans (scans made while sensor was steady)
static inline void load_map(PointCloudT::Ptr &map)
{
    PointCloudT::Ptr temp_(new PointCloudT);
    for (int i = 1; i < N_SCANS_MAP; ++i)
    {
        std::string mapfile = TEST_DATA_DIR "/map" + std::to_string(i) + ".pcd";
        std::cerr << "Loading file: " << mapfile << std::endl;

        if (pcl::io::loadPCDFile(mapfile, *temp_) != 0)
        {
            std::cerr << "Coud not load " << mapfile << "\n";
            // std::cerr << "Make sure you run the rest at the binaries folder.\n";
        }
        *map += *temp_;
    }

    std::cerr << "Loaded map: " << map->size() << " points.\n";
}


// Load scans during motion
static inline void load_scans(std::vector<PointCloudT::Ptr> &scans)
{

    for (int i = 1; i < N_SCANS_LASER; ++i)
    {
        PointCloudT::Ptr temp_(new PointCloudT);
        std::string scanfile = TEST_DATA_DIR "/scan" + std::to_string(i) + ".pcd";
        std::cerr << "Loading file: " << scanfile << std::endl;

        if (pcl::io::loadPCDFile(scanfile, *temp_) != 0)
        {
            std::cerr << "Coud not load " << scanfile << "\n";
            // std::cerr << "Make sure you run the rest at the binaries folder.\n";
        }

        scans.push_back(temp_);
    }

    std::cerr << "Loaded scans: " << scans.size() << "\n";
}

static inline void load_reference_map(PointCloudT::Ptr &reference_map)
{
    std::string refmapfile = TEST_DATA_DIR "/0_reference_map.pcd";
    std::cerr << "Loading file: " << refmapfile << std::endl;

    if (pcl::io::loadPCDFile(refmapfile, *reference_map) != 0)
    {
        exit(-1);
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

class SlamTest : public testing::Test
{
public:
    SlamTest()
    {
        // Load Map dataset
        m_scanmatch_map.reset(new PointCloudT);
        m_reference_map.reset(new PointCloudT);
        load_reference_map(m_reference_map);
        load_map(m_scanmatch_map);
        load_scans(scans);
    }

    virtual ~SlamTest() {}

protected:
    PointCloudT::Ptr m_scanmatch_map;
    PointCloudT::Ptr m_reference_map;
    std::vector<PointCloudT::Ptr> scans;

    double compareClouds(const PointCloudT::ConstPtr &reference, const PointCloudT::ConstPtr &source);
};

TEST_F(SlamTest, SeriesOfScanMatches)
{

    pcl::search::Search<PointT>::Ptr registration_seach;

    // Select method
    registration_seach.reset(new pcl::search::KdTree<PointT>);
    // registration_seach.reset(new pcl::search::Octree<PointT>(0.1));

    duna::RegistrationCost<3, PointT, PointT>::dataset_t data;
    duna::RegistrationCost<3, PointT, PointT> cost(&data);
    duna::Registration<3, PointT, PointT> registration(&cost);

    Eigen::Matrix4f guess = Eigen::Matrix4f::Identity(); // initial guess

    PointCloudT::Ptr hd_map(new PointCloudT);

    try
    {
        pcl::io::savePCDFileBinary("init_scanmatch_map.pcd", *m_scanmatch_map);
    }
    catch (pcl::IOException &ex)
    {
        std::cerr << ex.what() << std::endl;
        FAIL();
    }

    *hd_map = *m_scanmatch_map;

    for (int i = 0; i < scans.size(); ++i)
    {

        std::cerr << "SCAN #" << i << " ## " << std::endl;

        // Downsample
        auto start_time = std::chrono::high_resolution_clock::now();
        pcl::VoxelGrid<PointT> voxel_map;
        voxel_map.setInputCloud(m_scanmatch_map);
        voxel_map.setLeafSize(0.1, 0.1, 0.1);
        voxel_map.filter(*m_scanmatch_map);
        auto delta_time = std::chrono::high_resolution_clock::now() - start_time;
        std::cerr << "Voxel Grid Map : " << std::chrono::duration_cast<std::chrono::microseconds>(delta_time).count() << std::endl;

        start_time = std::chrono::high_resolution_clock::now();
        pcl::VoxelGrid<PointT> voxel_scan;
        voxel_scan.setInputCloud(scans[i]);
        voxel_scan.setLeafSize(0.1, 0.1, 0.1);
        PointCloudT::Ptr scan_processed(new PointCloudT);
        voxel_scan.filter(*scan_processed);
        delta_time = std::chrono::high_resolution_clock::now() - start_time;
        std::cerr << "Voxel Grid Scan : " << std::chrono::duration_cast<std::chrono::microseconds>(delta_time).count() << std::endl;

        // Compute normals
        start_time = std::chrono::high_resolution_clock::now();
        pcl::NormalEstimation<PointT, PointT> ne;
        // pcl::NormalEstimationOMP<PointT,PointT> ne;
        ne.setInputCloud(m_scanmatch_map);
        ne.setSearchMethod(registration_seach);
        ne.setKSearch(5);
        ne.compute(*m_scanmatch_map);
        delta_time = std::chrono::high_resolution_clock::now() - start_time;
        std::cerr << "Normal Estimation : " << std::chrono::duration_cast<std::chrono::microseconds>(delta_time).count() << std::endl;

        // Search tree rebuild/update
        start_time = std::chrono::high_resolution_clock::now();
        registration_seach->setInputCloud(m_scanmatch_map);
        delta_time = std::chrono::high_resolution_clock::now() - start_time;
        std::cerr << "Ocreee rebuild : " << std::chrono::duration_cast<std::chrono::microseconds>(delta_time).count() << std::endl;

        // Setup data;
        data.target = m_scanmatch_map;
        data.source = scan_processed;
        data.tgt_search_method = registration_seach;

        // Perform scan_matching
        start_time = std::chrono::high_resolution_clock::now();
        registration.setMaxCorrespondenceDistance(0.15);
        registration.setMaxIcpIterations(15); // 15
        registration.minimize(guess);
        delta_time = std::chrono::high_resolution_clock::now() - start_time;
        std::cerr << "Registration : " << std::chrono::duration_cast<std::chrono::microseconds>(delta_time).count() << std::endl;

        // Update guess
        guess = registration.getFinalTransformation();

        // Add new scan to map
        PointCloudT aligned_scan;
        pcl::transformPointCloud(*scans[i], aligned_scan, guess);

        *m_scanmatch_map = *m_scanmatch_map + aligned_scan;
        *hd_map = *hd_map + aligned_scan;
    }

    try
    {

        pcl::io::savePCDFileBinary("scanmatch_result.pcd", *hd_map);
        std::cerr << "Resulting map written to 'scanmatch_result.pcd' \n";
        std::cerr << "Initial map written to 'init_scanmatch_map.pcd' \n";
    }
    catch (pcl::IOException &ex)
    {
        std::cerr << ex.what() << std::endl;
        FAIL();
    }

    // Copare results
    std::cerr << "Checking results...\n";
    double total_error = compareClouds(m_reference_map, m_scanmatch_map);

    std::cerr << "Total error: " << total_error << std::endl;

    EXPECT_LE(total_error, 0.001);
}

double SlamTest::compareClouds(const PointCloudT::ConstPtr &reference, const PointCloudT::ConstPtr &source)
{
    double error = 0;
    pcl::search::KdTree<PointT> kdtree;

    kdtree.setInputCloud(reference);

    std::vector<int> indices_(1);
    std::vector<float> sqr_dists(1);

    for (const auto &it : source->points)
    {
        kdtree.nearestKSearch(it, 1, indices_, sqr_dists);
        // error += pcl::euclideanDistance(it,reference->points[indices_[0]]);
        // error += pcl::squaredEuclideanDistance(it, reference->points[indices_[0]]);
        error += sqr_dists[0];
    }
    return error / source->size();
}

// Total error: 0.0876619
