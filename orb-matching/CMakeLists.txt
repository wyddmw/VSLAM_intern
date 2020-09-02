cmake_minimum_required(VERSION 2.8)
project(orb_matching)

set(CMAKE_CXX_FLAGS "--std=c++11")
# 需要先找到对应的包 然后才能将路径添加进去
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDES_DIRS})

include_directories("/usr/include/eigen3")

# PCL
find_package( PCL REQUIRED COMPONENT common io )
include_directories( ${PCL_INCLUDE_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )

# g2o
#find_package(G2O REQUIRED)
include_directories("/home/spyder/g2o")

# Sophus
#find_package(Sophus REQUIRED)
include_directories("/home/spyder/Sophus")

# add_definitions(${PCL_DEFINITIONS})

# find_package(Pangolin REQUIRED)
add_executable(orb-matching orb-matching.cpp)
add_executable(pose pose_3d.cpp)
# target_link_libraries(stereo ${OpenCV_LIBS} ${PCL_LIBRARIES})
target_link_libraries(orb-matching ${OpenCV_LIBS} ${PCL_LIBRARIES} )
target_link_libraries(pose ${OpenCV_LIBS})
