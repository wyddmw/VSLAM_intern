cmake_minimum_required(VERSION 2.8)
project(image2cloud)

set(CMAKE_CXX_FLAGS "--std=c++11")
# 需要先找到对应的包 然后才能将路径添加进去
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDES_DIRS})

include_directories("/usr/include/eigen3")

# PCL
find_package( PCL REQUIRED COMPONENT common io )
include_directories( ${PCL_INCLUDE_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )

# add_definitions(${PCL_DEFINITIONS})

# find_package(Pangolin REQUIRED)
add_executable(stereo stereo_matching.cpp)
# target_link_libraries(stereo ${OpenCV_LIBS} ${PCL_LIBRARIES})
target_link_libraries( stereo ${OpenCV_LIBS} ${PCL_LIBRARIES} )