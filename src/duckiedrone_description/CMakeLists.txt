cmake_minimum_required(VERSION 3.0.2)
project(duckiedrone_description)

find_package(catkin REQUIRED)
catkin_package()

install(DIRECTORY urdf launch meshes
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
