#pragma once

#include <memory>
#include <duna/exception.h>

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
        virtual bool f(const Scalar *x, Scalar *f_x, unsigned int index) = 0;

        // Computes both jacobian and function at same time. Usually they depend on commons functions.
        virtual bool f_df(const Scalar *x, Scalar *f_x, Scalar *jacobian, unsigned int index) = 0;
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
        virtual bool f(const Scalar *x, Scalar *residual, unsigned int index) override = 0;

        // No jacobian definition.
        virtual bool f_df(const Scalar *x, Scalar *f_x, Scalar *jacobian, unsigned int index) final
        {
            return false;
        }
    };

    /* For jacobian defined models. */
    template <typename Scalar>
    class BaseModelJacobian : public IBaseModel<Scalar>
    {
    public:
        BaseModelJacobian() = default;
        virtual ~BaseModelJacobian() = default;

        // Setups up data for the model (i.e setting up transform 'T' from state vector 'x')
        virtual void setup(const Scalar *x) override {}

        // Update internal states of the model. (i.e registration correspondences)
        virtual void update(const Scalar *x) override {}

        // Function (r_i). Must return true if result if valid.
        virtual bool f(const Scalar *x, Scalar *f_x, unsigned int index) override
        {
            throw duna::Exception("Non implemented jacobian model function `f` being used.");
        }

        // No jacobian definition.
        virtual bool f_df(const Scalar *x, Scalar *f_x, Scalar *jacobian, unsigned int index) override = 0;
    };
} // namespace