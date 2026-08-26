cmake_minimum_required(VERSION 3.0.2)
project(duckiedrone_validation)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  gazebo_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/controllers/pid_controller.py
  scripts/controllers/mpc_controller.py
  scripts/controllers/vstmpc_controller.py
  scripts/scenarios/scenario_runner.py
  scripts/loggers/odom_bridge.py
  scripts/loggers/mixer_node.py
  scripts/controllers/pem_excitation_controller.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
