cmake_minimum_required(VERSION 3.10)

project(duna_optimizator VERSION 0.1 LANGUAGES CXX)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.10 REQUIRED COMPONENTS io common search registration)

link_libraries(Eigen3::Eigen)
include_directories(include src)

set (CMAKE_EXPORT_COMPILE_COMMANDS ON)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)

#set (CMAKE_BUILD_TYPE RELEASE)
#add_compile_options(-msse2 -msse3)

# duna libraries
file (GLOB duna_opt_src "src/*.cpp")
add_library(${PROJECT_NAME} SHARED ${duna_opt_src})
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES})



# add_executable(test_registration test/test_code_3.cpp)
# target_link_libraries(test_registration ${PROJECT_NAME})
# add_test(NAME test_registration0 COMMAND test_registration "bunny.pcd" WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/data)



get_filename_component(DATASET_PATH gtest/data/ REALPATH)
add_definitions(-DTEST_DATA_DIR="${DATASET_PATH}")

#gtest
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
enable_testing()
include(GoogleTest)


# Make test cases
file(GLOB test_src "gtest/*.cpp")
foreach(test_case ${test_src})
get_filename_component(test_src_name ${test_case} NAME_WE)
message(STATUS ${filename})    
add_executable(${test_src_name} ${test_case})
target_link_libraries(${test_src_name} ${PCL_LIBRARIES} ${PROJECT_NAME} gtest pthread)
gtest_discover_tests(${test_src_name})
endforeach()

# gtest_discover_tests(model_test)
# gtest_discover_tests(camera_calibration_test)
# gtest_discover_tests(registration_test)

# TODO auotmate tests execution after build
file(GLOB header_files "include/*" LIST_DIRECTORIES false)

message(STATUS ${header_files})
#set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${header_files}")
install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION lib
)


# TODO how to make sure client code will access header correctly?
install(DIRECTORY include/
        DESTINATION include
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
)