#pragma once

#include "duna/cost_function.hpp"
#include "duna/duna_log.h"
#include "duna/so3.h"

#include <limits>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <pcl/search/kdtree.h>
#include <pcl/correspondence.h>

// Error function
template <typename Scalar, typename PointTarget>
inline Scalar computeError(const Eigen::Matrix<Scalar, 4, 1> &warped_src_pt, const PointTarget &tgt_pt)
{
    Eigen::Matrix<Scalar,4,1> tgt(tgt_pt.x, tgt_pt.y, tgt_pt.z, 0);
    Eigen::Matrix<Scalar,4,1> src(warped_src_pt[0], warped_src_pt[1], warped_src_pt[2], 0);
    return (src - tgt).norm(); // TODO norm vs norm² ?
}

// Error function
template <typename Scalar>
inline Scalar computeError(const Eigen::Matrix<Scalar, 4, 1> &warped_src_pt, const pcl::PointNormal &tgt_pt)
{
    Eigen::Matrix<Scalar,4,1> tgt(tgt_pt.x, tgt_pt.y, tgt_pt.z, 0);
    Eigen::Matrix<Scalar,4,1> src(warped_src_pt[0], warped_src_pt[1], warped_src_pt[2], 0);
    Eigen::Matrix<Scalar,4,1> tgt_normal(tgt_pt.normal_x, tgt_pt.normal_y, tgt_pt.normal_z,0);
    return (src - tgt).dot(tgt_normal); // TODO norm vs norm² ?
}

template <int NPARAM, typename PointSource, typename PointTarget>
class RegistrationCost : public CostFunction<NPARAM>
{
public:
    struct dataset_t
    {
        typename pcl::PointCloud<PointSource>::ConstPtr source;
        typename pcl::PointCloud<PointTarget>::ConstPtr target;
        typename pcl::search::KdTree<PointTarget>::ConstPtr tgt_kdtree;
    };

    using VectorN = typename CostFunction<NPARAM>::VectorN;
    using VectorNd = Eigen::Matrix<double, NPARAM, 1>;
    using MatrixN = typename CostFunction<NPARAM>::MatrixN;

    using Matrix4 = Eigen::Matrix4f;

    using PointCloudSourceConstPtr = typename pcl::PointCloud<PointSource>::ConstPtr;

    using CostFunction<NPARAM>::m_dataset;

    RegistrationCost(void *dataset) : CostFunction<NPARAM>(dataset)
    {
        l_dataset = reinterpret_cast<dataset_t *>(m_dataset);

        if (l_dataset->source == nullptr || l_dataset->target == nullptr || l_dataset->tgt_kdtree == nullptr)
        {
            throw std::runtime_error("Invalid dataset. Check if dataset pointers are allocated.\n");
        }

        // Checkdataset
        DUNA_DEBUG("source pts : %ld, tgt pts: %ld\n", l_dataset->source->size(), l_dataset->target->size());
        // TODO check KDTREE
    }

  
    virtual ~RegistrationCost() = default;

    // TODO implement
    void checkData() override
    {
    }

    
    inline void setTransformedSourcePtr(PointCloudSourceConstPtr transformed_src)
    {
        m_transformed_source = transformed_src;
    }

    inline void setCorrespondencesPtr(pcl::CorrespondencesConstPtr correspondences)
    {
        m_correspondences = correspondences;
    }

    double computeCost(const VectorN &x) override
    {

        Eigen::Matrix4f transform;
        so3::param2Matrix(x, transform);

        double sum = 0;
        for (int i = 0; i < m_correspondences->size(); ++i)
        {
            const PointSource &src_pt = m_transformed_source->points[(*m_correspondences)[i].index_query];
            const PointTarget &tgt_pt = l_dataset->target->points[(*m_correspondences)[i].index_match];

            const Eigen::Vector4f &src_pt_vec = src_pt.getVector4fMap();

            const Eigen::Vector4f src_pt_warped_vec = transform * src_pt_vec;

            double xout = computeError(src_pt_warped_vec, tgt_pt);

            sum += xout * xout;
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

        // TODO we're having all kinds of numeric errors here :(. Usually we want smallest possible without breaking numeric stability.
        const float epsilon = 12 * (std::numeric_limits<float>::epsilon());
        float h = epsilon;
        for (int j = 0; j < NPARAM; ++j)
        {
            // float h = epsilon * abs(x[j]);
            // if (h == 0.)
            //  h = epsilon;

            VectorN x_plus(x);
            // VectorNd x_minus(x_double);

            x_plus[j] += h;
            // x_minus[j] -= epsilon;

            so3::param2Matrix(x_plus, transform_plus[j]);
            // so3::param2Matrix(x_minus, transform_minus[j]);
        }

        double sum = 0;
        for (int i = 0; i < m_correspondences->size(); ++i)
        {
            const PointSource &src_pt = m_transformed_source->points[(*m_correspondences)[i].index_query];
            const PointTarget &tgt_pt = l_dataset->target->points[(*m_correspondences)[i].index_match];

            const Eigen::Vector4f &src_pt_vec = src_pt.getVector4fMap();

            const Eigen::Vector4f src_pt_warped_vec = transform * src_pt_vec;

            double xout = computeError(src_pt_warped_vec, tgt_pt);

            for (int j = 0; j < NPARAM; ++j)
            {
                // float h = epsilon * abs(x[j]);
                // if (h == 0.)
                //  h = epsilon;

                Eigen::Vector4f src_pt_warped_plus_vec = transform_plus[j] * src_pt_vec;
                // Eigen::Vector4d src_pt_warped_minus_vec = transform_minus[j] * src_pt_vec.template cast<double>();

                double xout_plus = computeError(src_pt_warped_plus_vec, tgt_pt);
                // double xout_minus = computeError(src_pt_warped_minus_vec, tgt_pt);

                // TODO this is numerically unstable
                jacobian_row[j] = (xout_plus - xout) / (h);
            }

            hessian.template selfadjointView<Eigen::Lower>().rankUpdate(jacobian_row.transpose()); // this sums ? yes
            // hessian += jacobian_row.transpose()*jacobian_row;
            b += jacobian_row.transpose() * (xout);

            sum += xout * xout;
        }

        // Crazingly engouth, we may comment this
        hessian.template triangularView<Eigen::Upper>() = hessian.transpose();

        return sum;
    }

private:
    pcl::CorrespondencesConstPtr m_correspondences;
    PointCloudSourceConstPtr m_transformed_source;
    dataset_t *l_dataset; // cast

    
};