#ifndef REGISTRATION_H
#define REGISTRATION_H

#include <duna/registration/registration_base.h>
#include <duna/cost_function_numerical.h>
#include <duna/cost_function_analytical.h>
#include <pcl/common/transforms.h>
#include <duna/logging.h>
#include <duna/registration/registration_model.h>

namespace duna
{

    template <typename PointSource, typename PointTarget, typename Scalar = double>
    class Registration : public RegistrationBase<PointSource, PointTarget, Scalar>
    {
        using Matrix4 = typename RegistrationBase<PointSource, PointTarget, Scalar>::Matrix4;
        using RegistrationBase<PointSource, PointTarget, Scalar>::m_max_icp_iterations;
        using RegistrationBase<PointSource, PointTarget, Scalar>::m_correspondences;
        using RegistrationBase<PointSource, PointTarget, Scalar>::m_transformed_source;
        using RegistrationBase<PointSource, PointTarget, Scalar>::m_source;
        using RegistrationBase<PointSource, PointTarget, Scalar>::m_target;
        using RegistrationBase<PointSource, PointTarget, Scalar>::m_max_corr_dist;
        using RegistrationBase<PointSource, PointTarget, Scalar>::m_target_search_method;
        using RegistrationBase<PointSource, PointTarget, Scalar>::m_nearest_k;
        using RegistrationBase<PointSource, PointTarget, Scalar>::m_final_transformation;
        using RegistrationBase<PointSource, PointTarget, Scalar>::m_optimizer;

    public:
        void align() override
        {
            m_final_transformation.setIdentity();
            align(m_final_transformation);
        }

        void align(const Matrix4 &guess) override
        {
            if(!m_target_search_method)
                throw std::runtime_error("No KD found");
            
            if(!m_target)
                throw std::runtime_error("No Target Point Cloud found");

            if(!m_source)
                throw std::runtime_error("No Source Point Cloud found");



            DUNA_DEBUG("Target pts: %ld \n", m_target->size());
            DUNA_DEBUG("Source pts: %ld \n", m_source->size());
            registrationLoop();
        }


    protected:
        void updateCorrespondences() override
        {
            m_correspondences.clear();

            std::vector<float> sqrd_distances(m_nearest_k);
            std::vector<int> indices(m_nearest_k);

            float max_sqrd_dist = m_max_corr_dist * m_max_corr_dist;
            for (int i = 0; i < m_transformed_source->size(); ++i)
            {
                const PointSource &queryPoint = m_transformed_source->points[i];

                m_target_search_method->nearestKSearchT(queryPoint, m_nearest_k, indices, sqrd_distances);

                if (sqrd_distances[0] > max_sqrd_dist)
                    continue;

                // TODO precompute normals
                pcl::Correspondence corr;
                corr.index_query = i;
                corr.index_match = indices[0];
                m_correspondences.push_back(corr);
            }

            if (m_correspondences.size() == 0)
                throw std::runtime_error("No correspondences found.");
        }

        void registrationLoop()
        {

            pcl::transformPointCloud(*m_source,*m_transformed_source, m_final_transformation);

            Point2Point<PointSource, PointTarget, Scalar> model(*m_transformed_source, *m_target, m_correspondences);
            duna::CostFunctionNumericalDiff<Point2Point<PointSource, PointTarget, Scalar>,Scalar,6,1> cost(&model);
            // duna::CostFunctionAnalytical<Point2Point<PointSource, PointTarget, Scalar>,Scalar,6,1> cost(&model);

            m_optimizer->setCost(&cost);

            Eigen::Matrix<Scalar,6,1> x0;

            Matrix4 delta_transform;

            for (int i = 0; i < m_max_icp_iterations; ++i)
            {
                DUNA_DEBUG("ICP ITERATION #%d / %d \n", i+1 , m_max_icp_iterations);
                updateCorrespondences();

                cost.setNumResiduals(m_correspondences.size());

                x0.setZero();
                OptimizationStatus status = m_optimizer->minimize(x0.data());

                    if(status == OptimizationStatus::NUMERIC_ERROR)
                        throw std::runtime_error("Numeric error");

                
                so3::convert6DOFParameterToMatrix(x0.data(), delta_transform);

                pcl::transformPointCloud(*m_transformed_source,*m_transformed_source, delta_transform);

                m_final_transformation = delta_transform * m_final_transformation;
            
            }


        }
    };
}

#endif