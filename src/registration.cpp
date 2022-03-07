#include <duna/impl/registration.hpp>



template class Registration<6,pcl::PointXYZ,pcl::PointXYZ>;
template class Registration<6,pcl::PointXYZ,pcl::PointNormal>;

template class Registration<3,pcl::PointXYZ,pcl::PointXYZ>;
template class Registration<3,pcl::PointXYZ,pcl::PointNormal>;