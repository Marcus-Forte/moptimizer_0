#pragma once

#include <Eigen/Dense>
#include <memory>


namespace duna::covariance
{
    // Covariance function interface.
    template <typename T, int DIM = Eigen::Dynamic>
    class ICovariance
    {
        public:
        using Ptr = std::shared_ptr<ICovariance>;
        using ConstPtr = std::shared_ptr<const ICovariance>;
        using MatrixType = Eigen::Matrix<T, DIM, DIM>;
        
        ICovariance() = default;
        virtual ~ICovariance() = default;

        virtual MatrixType getCovariance(T* input = 0) = 0;
    };

    
    template <typename T>
    class NoCovariance : public ICovariance<T, 1>
    {
        public:
        using typename ICovariance<T,1>::MatrixType;

        inline MatrixType getCovariance(T* input = 0) override
        {
            return MatrixType::Identity();
        }

    };
}