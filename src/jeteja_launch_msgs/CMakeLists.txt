cmake_minimum_required(VERSION 3.8)
project(jeteja_launch_msgs)

# Compiler warnings
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)

# Specify message files
set(msg_files
  "msg/PwmSignals.msg"
  "msg/PreprocessedImage.msg"
)

# Generate interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES builtin_interfaces sensor_msgs
)

# Export runtime dependencies
ament_export_dependencies(rosidl_default_runtime)
ament_export_dependencies(builtin_interfaces)

ament_package()
