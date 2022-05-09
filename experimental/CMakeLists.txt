cmake_minimum_required(VERSION 3.10)
project(experimental)


find_package(OpenCV REQUIRED)


add_executable(scene_reconstruct src/scene_reconstruct.cpp)
target_include_directories(scene_reconstruct PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(scene_reconstruct ${OpenCV_LIBRARIES})