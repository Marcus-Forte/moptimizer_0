#pragma once

#include <pcl/registration/correspondence_rejection.h>
#include <pcl/registration/correspondence_estimation.h>

namespace duna
{
    template <typename PointSource, typename PointTarget, typename Scalar = double>
    class Registration
    {
    public:
        Registration() = default;
        virtual ~Registration() = default;


        void align();

    protected:
        pcl::registration::CorrespondenceEstimation<PointSource, PointTarget, Scalar> estimator_;
    };
} // namespace