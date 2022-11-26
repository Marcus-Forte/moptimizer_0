#pragma once

#include <memory>

namespace duna
{
    /* Interface definitions for user models.*/

    template <typename Scalar>
    class IBaseModel
    {
    public:
        using Ptr = std::shared_ptr<IBaseModel>;
        using ConstPtr = std::shared_ptr<const IBaseModel>;
        IBaseModel() = default;
        virtual ~IBaseModel() = default;

        // Setups up data for the model (i.e setting up transform 'T' from state vector 'x')
        virtual void setup(const Scalar *x) = 0;

        // Update internal states of the model. (i.e registration correspondences)
        virtual void update(const Scalar *x) = 0;

        // Function (r_i)
        virtual bool operator()(const Scalar *x, Scalar *residual, unsigned int index) = 0;

        // Jacobian (J_i), row major.
        virtual void df(const Scalar *x, Scalar *jacobian, unsigned int index) = 0;
    };

    /* For non-jacobian defined models. */
    template <typename Scalar>
    class BaseModel : public IBaseModel<Scalar>
    {
    public:
        BaseModel() = default;
        virtual ~BaseModel() = default;

        // Setups up data for the model (i.e setting up transform 'T' from state vector 'x')
        virtual void setup(const Scalar *x) override {}

        // Update internal states of the model. (i.e registration correspondences)
        virtual void update(const Scalar *x) override {}

        // Function (r_i). Must return true if result if valid.
        virtual bool operator()(const Scalar *x, Scalar *residual, unsigned int index) override = 0;

        // No jacobian definition.
        void df(const Scalar *x, Scalar *jacobian, unsigned int index) override {}
    };

    /* For jacobian defined models. */
    template <typename Scalar>
    class BaseModelJacobian : public BaseModel<Scalar>
    {
    public:
        BaseModelJacobian() = default;
        virtual ~BaseModelJacobian() = default;

        // Jacobian (J_i), row major.
        virtual void df(const Scalar *x, Scalar *jacobian, unsigned int index) override = 0;
    };
} // namespace