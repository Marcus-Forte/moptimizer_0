#pragma once

#include <memory>

namespace duna
{
    /* Interface definition for user models.*/
    template <typename Scalar>
    class BaseModel
    {
    public:
        using Ptr = std::shared_ptr<BaseModel>;
        using ConstPtr = std::shared_ptr<const BaseModel>;
        BaseModel() = default;
        virtual ~BaseModel() = default;

        virtual void init(const Scalar *x) {}
        virtual void setup(const Scalar *x) {}

        // Function
        virtual void operator()(const Scalar *x, Scalar *residual, unsigned int index) = 0;
    };

    template <typename Scalar>
    class BaseModelJacobian : public BaseModel<Scalar>
    {
    public:
        using Ptr = std::shared_ptr<BaseModelJacobian>;
        using ConstPtr = std::shared_ptr<const BaseModelJacobian>;
        BaseModelJacobian() = default;
        virtual ~BaseModelJacobian() = default;

        // Jacobian
        virtual void df(const Scalar *x, Scalar *residual, unsigned int index) = 0;
    };
} // namespace