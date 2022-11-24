#pragma once

#include <duna/registration/scan_matching.h>
#include <duna/cost_function_analytical.h>
#include <duna/registration/models/point2plane3dof.h>
#include <duna/levenberg_marquadt.h>

namespace duna
{
    /* This class is a thin wrapper around the optimization object specific for point cloud registration / scan matching */
    template <typename PointSource, typename PointTarget, typename Scalar = double>
    class ScanMatching3DOF : public ScanMatchingBase<PointSource, PointTarget, Scalar>
    {
    public:
        ScanMatching3DOF()
        {
            optimizer_.reset(new duna::LevenbergMarquadt<Scalar, 3>);
            optimizer_->getLogger().setVerbosityLevel(L_DEBUG);
            logger_.setLoggerName("Matcher3DOF");
        }
        virtual ~ScanMatching3DOF() = default;

        void match(Scalar *x0) override;

        

    private:
        using PointCloudSource = typename ScanMatchingBase<PointSource, PointTarget, Scalar>::PointCloudSource;
        using PointCloudSourcePtr = typename ScanMatchingBase<PointSource, PointTarget, Scalar>::PointCloudSourcePtr;
        using ScanMatchingBase<PointSource, PointTarget, Scalar>::optimizer_;
        using ScanMatchingBase<PointSource, PointTarget, Scalar>::corr_estimator_;
        using ScanMatchingBase<PointSource, PointTarget, Scalar>::target_tree_;
        using ScanMatchingBase<PointSource, PointTarget, Scalar>::logger_;
        using ScanMatchingBase<PointSource, PointTarget, Scalar>::source_;
        using ScanMatchingBase<PointSource, PointTarget, Scalar>::target_;
        using ScanMatchingBase<PointSource, PointTarget, Scalar>::max_corr_distance_;
        using ScanMatchingBase<PointSource, PointTarget, Scalar>::max_num_iterations_;
        using ScanMatchingBase<PointSource, PointTarget, Scalar>::max_num_opt_iterations_;
        using ScanMatchingBase<PointSource, PointTarget, Scalar>::final_transform_;
    };
} // namespace