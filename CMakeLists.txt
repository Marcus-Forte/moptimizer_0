cmake_minimum_required(VERSION 3.10)

project(duna_optimizer VERSION 0.1 LANGUAGES CXX)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.10 REQUIRED COMPONENTS io common search)

link_libraries(Eigen3::Eigen)
include_directories(include src)

set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

# duna libraries
file (GLOB duna_opt_src "src/*.cpp")
add_library(${PROJECT_NAME} SHARED ${duna_opt_src})
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES})



# add_executable(test_registration test/test_code_3.cpp)
# target_link_libraries(test_registration ${PROJECT_NAME})
# add_test(NAME test_registration0 COMMAND test_registration "bunny.pcd" WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/data)

#gtest
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

enable_testing()

# TODO implement test class  
add_executable(
    model_test
    gtest/model_test.cpp
)


target_link_libraries(model_test
    ${PROJECT_NAME}
    gtest
    pthread
)

add_executable(
    camera_calibration_test
    gtest/camera_calibration_test.cpp
)

target_link_libraries(camera_calibration_test
    ${PROJECT_NAME}
    gtest
    pthread
)


include(GoogleTest)
gtest_discover_tests(model_test)
gtest_discover_tests(camera_calibration_test)

# TODO auotmate tests execution after build