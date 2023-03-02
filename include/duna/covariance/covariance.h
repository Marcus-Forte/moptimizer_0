#pragma once

#include <Eigen/Dense>
#include <memory>
#include "duna/types.h"

namespace duna::covariance
{
    // Covariance function interface.
    template <typename T>
    class ICovariance
    {
    public:
        using Ptr = std::shared_ptr<ICovariance>;
        using ConstPtr = std::shared_ptr<const ICovariance>;
        using MatrixType = Eigen::Matrix<T, duna::Dynamic, duna::Dynamic>;

        ICovariance() = default;
        virtual ~ICovariance() = default;

        virtual MatrixType getCovariance(T *input = 0) = 0;
    };

    /* No Covariance / Identity Covariance*/
    template <typename T>
    class IdentityCovariance : public ICovariance<T>
    {
    public:
        using typename ICovariance<T>::MatrixType;
        /* Identity covariance dimension. */
        IdentityCovariance(unsigned int dimension);
        virtual ~IdentityCovariance();
        MatrixType getCovariance(T *input = 0) override;

        protected:
        MatrixType covariance_matrix_;
    };
}