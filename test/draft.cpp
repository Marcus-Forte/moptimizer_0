#include <gtest/gtest.h>
#include <duna/model.h>

using namespace duna;

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

class ModelChild : public Model
{
public:
    ModelChild(const int var) : member(var)
    {
        std::cout << "ModelChild Ctr\n";
    }

    ModelChild(const ModelChild &rhs)
    {
        std::cout << "ModelChild CopyCtr\n";
    }
    virtual ~ModelChild()
    {
        std::cout << "ModelChild DECtr\n";
    }
    /* Client Code can use this method for settig up the cost computation (i.e convert parameter X into transform matrix) */
    inline virtual void setup(const float *x)
    {
        std::cout << member << std::endl;
    };

    /*  Evaluate cost function at index */
    inline virtual void operator()(const float *x, float *f_x, const unsigned int index)
    {
    }

    void setX(float x){
        member = x;
    }

    GENERATE_CLONE(ModelChild);

public:
    float member = 0.0;
};

TEST(Polymorphism, DerivedCopy)
{
    Model<ModelChild>* parent;

    parent = new ModelChild(1.0);

    parent->setup(0);        

    std::vector<Model<ModelChild>*> parent_vector;
    parent_vector.push_back(parent->clone(15));

    parent->setX(10);

    parent_vector[0]->setup(0);

    
    
    
}