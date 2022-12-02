cmake_minimum_required(VERSION 3.10)
project(experimental)

include_directories(../include)

find_package(duna_optimizer REQUIRED PATHS /home/marcus/workspace/nomad/duna-optimizer/build/install)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED PATHS /home/marcus/Workspace/coding/cpp/ceres-solver/build/install)

link_libraries(${OpenCV_LIBRARIES} Ceres::ceres)
# add_executable(scene_reconstruct src/scene_reconstruct.cpp)
# target_include_directories(scene_reconstruct PUBLIC ${OpenCV_INCLUDE_DIRS})


add_executable(ceres_helloworld src/ceres_helloworld.cpp)
add_executable(ceres_camera_calibration src/ceres_camera_calibration.cpp)

target_link_libraries(ceres_camera_calibration duna::duna_optimizer)

