#pragma once

#include "cost_function.hpp"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/correspondence.h>
#include <pcl/common/transforms.h>

#include <assert.h>
/*

/* Define your dataset */
struct datatype_t
{
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr source;
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr target;
    pcl::search::KdTree<pcl::PointXYZ>::ConstPtr tgt_kdtree;
};

template <int NPARAM>
class RegistrationCost : public CostFunction<NPARAM>
{
public:
    using VectorN = typename CostFunction<NPARAM>::VectorN;
    using MatrixN = typename CostFunction<NPARAM>::MatrixN;
    using VectorX = typename CostFunction<NPARAM>::VectorX;
    using MatrixX = typename CostFunction<NPARAM>::MatrixX;
    using Matrix4 = Eigen::Matrix4f;

    using CostFunction<NPARAM>::m_dataset;
    using CostFunction<NPARAM>::m_data_size;

    RegistrationCost(unsigned int data_size, void *dataset) : CostFunction<NPARAM>(data_size, dataset)
    {
        datatype_t *l_dataset = reinterpret_cast<datatype_t *>(m_dataset);
        m_source_transformed = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    }
    virtual ~RegistrationCost() = default;

    inline void setMaxCorrDist(float dist)
    {
        m_max_correspondence_dist = dist;
    }

    void init(const VectorN &x0)
    {

        datatype_t *l_dataset = reinterpret_cast<datatype_t *>(m_dataset);

        Matrix4 transform_matrix_;
        param2Matrix(x0, transform_matrix_);

        // move point cloud
        pcl::transformPointCloud(*l_dataset->source, *m_source_transformed, transform_matrix_);
    }

    void preprocess(const VectorN &x0) override
    {
        datatype_t *l_dataset = reinterpret_cast<datatype_t *>(m_dataset);

        m_correspondences.clear();
        m_correspondences.reserve(m_source_transformed->size());

        pcl::Indices indices(m_k_neighboors);
        std::vector<float> k_distances(m_k_neighboors);

        // compute correspondences
        for (int i = 0; i < m_source_transformed->size(); ++i)
        {
            const pcl::PointXYZ &pt = m_source_transformed->points[i];

            l_dataset->tgt_kdtree->nearestKSearch(pt, m_k_neighboors, indices, k_distances);

            if (k_distances[0] > m_max_correspondence_dist)
                continue;

            // Compute normal

            pcl::Correspondence correspondence;
            correspondence.index_match = indices[0];
            correspondence.index_query = i;
            m_correspondences.push_back(correspondence);
        }

        DUNA_DEBUG("source pts : %ld, corr pts: %ld", m_source_transformed->size(), m_correspondences.size());

        if (m_correspondences.size() == 0)
        {
            throw std::runtime_error("no more correspondences.");
        }
    }

    void postprocess(const VectorN &x0) override
    {
        Matrix4 transform_matrix_;
        param2Matrix(x0, transform_matrix_);

        // move point cloud
        pcl::transformPointCloud(*m_source_transformed, *m_source_transformed, transform_matrix_);
    }

    double computeCost(const VectorN &x) override
    {
        double sum = 0;

        return sum;
    }

    double linearize(const VectorN &x, MatrixN &hessian, VectorN &b) override
    {
        double sum = 0;

        return sum;
    }

private:
    float m_max_correspondence_dist = 1.0;
    unsigned int m_k_neighboors = 5;
    pcl::Correspondences m_correspondences;
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_source_transformed;

    void param2Matrix(const VectorN &x0, Matrix4 &transform_matrix_)
    {
        // Matrix from vector
        transform_matrix_.setZero();
        transform_matrix_(0, 3) = x0[0];
        transform_matrix_(1, 3) = x0[1];
        transform_matrix_(2, 3) = x0[2];
        transform_matrix_(3, 3) = 1;

        // Compute w from the unit quaternion
        Eigen::Quaternion<float> q(0, x0[3], x0[4], x0[5]);
        q.w() = static_cast<float>(std::sqrt(1 - q.dot(q)));
        q.normalize();
        transform_matrix_.topLeftCorner(3, 3) = q.toRotationMatrix();
    }
};