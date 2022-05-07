#ifndef MODEL_H
#define MODEL_H

namespace duna 
{
    template <class Scalar>
    class Model
    {
        public:
        Model() = default;
        virtual ~Model() = default;

        /* Client Code can use this method for settig up the cost computation (i.e convert parameter X into transform matrix) */
        inline virtual void setup(const Scalar* x) {};

        /*  Evaluate cost function at index */
        inline virtual void operator()(const Scalar* x, Scalar* f_x, const unsigned int index) = 0;



    };
}
#endif