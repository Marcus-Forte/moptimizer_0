#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <pcl/registration/correspondence_estimation.h>
#include <memory>
/* Use this file to do quick prototyping. */

class Base{
  public:
  virtual void perform() const {
    std::cout << "Base Perform\n";
  }
};

class Child : public Base{
  public:
   void perform() const override {
    std::cout << "Child Perform\n";
  }
};


void fun(const std::shared_ptr<const Child>& input ) {
  input->perform();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}

TEST(Draft, Draft1) {
   Base * base = new Child;

  auto cast = dynamic_cast<Child*>(base);

  if(cast == nullptr) {
    std::cout << "unable to cast.\n";
    exit(-1);
  }
  cast->perform();

  // base = new Base;

  Base*  cast_ = (Base*)(base);

  cast_->perform();

  // fun dynamic_cast<std::shared_ptr<Child>>(std::make_shared<Base>()) );
}