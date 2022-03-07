#pragma once

#include "duna/cost_function.hpp"
#include "duna/so3.h"

#include <limits>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <pcl/search/kdtree.h>
#include <pcl/correspondence.h>

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
    using VectorNd = Eigen::Matrix<double, NPARAM, 1>;
    using MatrixN = typename CostFunction<NPARAM>::MatrixN;

    using Matrix4 = Eigen::Matrix4f;

    using CostFunction<NPARAM>::m_dataset;

    RegistrationCost(void *dataset) : CostFunction<NPARAM>(dataset)
    {
        l_dataset = reinterpret_cast<datatype_t *>(m_dataset);
        
        if (l_dataset->source == nullptr || l_dataset->target == nullptr || l_dataset->tgt_kdtree == nullptr)
        {
            throw std::runtime_error("Invalid dataset. Check if dataset pointers are allocated.\n");
        }

        // Checkdataset
        DUNA_DEBUG("source pts : %ld, tgt pts: %ld\n", l_dataset->source->size(), l_dataset->target->size());
        // TODO check KDTREE
    }

    // TODO check numerical stability
    virtual ~RegistrationCost() = default;

    // TODO implement
    void checkData() override
    {
    }

    inline void setTransformedSourcePtr(pcl::PointCloud<pcl::PointXYZ>::ConstPtr transformed_src)
    {
        m_transformed_source = transformed_src;
    }

    inline void setCorrespondencesPtr(pcl::CorrespondencesConstPtr correspondences)
    {
        m_correspondences = correspondences;
    }

    double computeCost(const VectorN &x) override
    {

        VectorNd x_double(x.template cast<double>());
        Eigen::Matrix4d transform;
        so3::param2Matrix(x_double, transform);

        double sum = 0;
        for (int i = 0; i < m_correspondences->size(); ++i)
        {
            const pcl::PointXYZ &src_pt = m_transformed_source->points[(*m_correspondences)[i].index_query];
            const pcl::PointXYZ &tgt_pt = l_dataset->target->points[(*m_correspondences)[i].index_match];

            const Eigen::Vector4f &src_pt_vec = src_pt.getVector4fMap();

            const Eigen::Vector4d src_pt_warped_vec = transform * src_pt_vec.template cast<double>();

            double xout = computeError(src_pt_warped_vec, tgt_pt);

            sum += xout;
        }
        return sum;
    }

    double linearize(const VectorN &x, MatrixN &hessian, VectorN &b) override
    {

        hessian.setZero();
        b.setZero();

        Eigen::Matrix<float, 1, NPARAM> jacobian_row;
        Eigen::Matrix4f transform_plus[NPARAM];
        // Eigen::Matrix4d transform_minus[NPARAM];
        Eigen::Matrix4f transform;

        so3::param2Matrix(x, transform);

        // TODO we're having all kinds of numeric errors here :(. Usually we want smallest possible without breaking
        float epsilon = std::numeric_limits<float>::epsilon();

        for (int j = 0; j < NPARAM; ++j)
        {
            VectorN x_plus(x);
            // VectorNd x_minus(x_double);

            x_plus[j] += epsilon;
            // x_minus[j] -= epsilon;

            so3::param2Matrix(x_plus, transform_plus[j]);
            // so3::param2Matrix(x_minus, transform_minus[j]);
        }

        double sum = 0;
        for (int i = 0; i < m_correspondences->size(); ++i)
        {
            const pcl::PointXYZ &src_pt = m_transformed_source->points[(*m_correspondences)[i].index_query];
            const pcl::PointXYZ &tgt_pt = l_dataset->target->points[(*m_correspondences)[i].index_match];

            const Eigen::Vector4f &src_pt_vec = src_pt.getVector4fMap();

            const Eigen::Vector4f src_pt_warped_vec = transform * src_pt_vec;

            double xout = computeError(src_pt_warped_vec, tgt_pt);

            for (int j = 0; j < NPARAM; ++j)
            {

                Eigen::Vector4f src_pt_warped_plus_vec = transform_plus[j] * src_pt_vec;
                // Eigen::Vector4d src_pt_warped_minus_vec = transform_minus[j] * src_pt_vec.template cast<double>();

                double xout_plus = computeError(src_pt_warped_plus_vec, tgt_pt);
                // double xout_minus = computeError(src_pt_warped_minus_vec, tgt_pt);

                // TODO this is numerically unstable
                jacobian_row[j] = (xout_plus - xout) / (epsilon);
            }

            hessian.template selfadjointView<Eigen::Lower>().rankUpdate(jacobian_row.transpose()); // this sums ? yes
            // hessian += jacobian_row.transpose()*jacobian_row;
            b += jacobian_row.transpose() * (xout);

            sum += xout;
        }

        // hessian.template triangularView<Eigen::Upper>() = hessian.transpose();

        return sum;
    }

private:
    pcl::CorrespondencesConstPtr m_correspondences;
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr m_transformed_source;
    datatype_t *l_dataset; // cast

    template <typename Scalar>
    inline double computeError(const Eigen::Matrix<Scalar, 4, 1> &warped_src_pt, const pcl::PointXYZ &tgt_pt)
    {
        Eigen::Vector4f tgt(tgt_pt.x, tgt_pt.y, tgt_pt.z, 0);
        Eigen::Vector4f src(warped_src_pt[0], warped_src_pt[1], warped_src_pt[2], 0);
        return (src - tgt).norm(); // TODO norm vs norm² ?
    }
};