#ifndef MODEL_H
#define MODEL_H

namespace duna 
{
    template <class Scalar = double>
    class Model
    {
        public:
        Model() = default;
        virtual ~Model() = default;

        inline virtual void setup(const Scalar* x) = 0;
        inline virtual void computeAtIndex(const Scalar* x, Scalar* f_x, const unsigned int index) = 0;

    };
}
#endif