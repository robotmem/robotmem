cmake_minimum_required(VERSION 3.8)
project(robotmem_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Memory.msg"
  "msg/PerceptionData.msg"
  "msg/NodeStatus.msg"
  "srv/Learn.srv"
  "srv/Recall.srv"
  "srv/SavePerception.srv"
  "srv/Forget.srv"
  "srv/Update.srv"
  "srv/StartSession.srv"
  "srv/EndSession.srv"
  DEPENDENCIES std_msgs
)

ament_package()
