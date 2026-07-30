cmake_minimum_required(VERSION 3.8)
project(unitree_g1d_driver)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Threads REQUIRED)

set(UNITREE_SDK2_ROOT "" CACHE PATH "Path to the Unitree SDK2 source tree")
if(NOT UNITREE_SDK2_ROOT AND DEFINED ENV{UNITREE_SDK2_ROOT})
  set(UNITREE_SDK2_ROOT "$ENV{UNITREE_SDK2_ROOT}")
endif()
if(NOT UNITREE_SDK2_ROOT)
  get_filename_component(
    UNITREE_SDK2_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../unitree_sdk2"
    ABSOLUTE
  )
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(amd64|AMD64)$")
  set(UNITREE_SDK2_ARCH "x86_64")
else()
  set(UNITREE_SDK2_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(UNITREE_SDK2_LIBRARY
  "${UNITREE_SDK2_ROOT}/lib/${UNITREE_SDK2_ARCH}/libunitree_sdk2.a"
)
set(UNITREE_DDS_LIBRARY_DIR
  "${UNITREE_SDK2_ROOT}/thirdparty/lib/${UNITREE_SDK2_ARCH}"
)
set(UNITREE_DDS_C_LIBRARY "${UNITREE_DDS_LIBRARY_DIR}/libddsc.so")
set(UNITREE_DDS_CXX_LIBRARY "${UNITREE_DDS_LIBRARY_DIR}/libddscxx.so")

foreach(required_path
    "${UNITREE_SDK2_ROOT}/include"
    "${UNITREE_SDK2_ROOT}/thirdparty/include"
    "${UNITREE_SDK2_LIBRARY}"
    "${UNITREE_DDS_C_LIBRARY}"
    "${UNITREE_DDS_CXX_LIBRARY}")
  if(NOT EXISTS "${required_path}")
    message(FATAL_ERROR
      "Unitree SDK2 dependency is missing: ${required_path}. "
      "Set -DUNITREE_SDK2_ROOT=/path/to/unitree_sdk2."
    )
  endif()
endforeach()

add_library(unitree_sdk2_vendor STATIC IMPORTED GLOBAL)
set_target_properties(unitree_sdk2_vendor PROPERTIES
  IMPORTED_LOCATION "${UNITREE_SDK2_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES
    "${UNITREE_SDK2_ROOT}/include;${UNITREE_SDK2_ROOT}/thirdparty/include;${UNITREE_SDK2_ROOT}/thirdparty/include/ddscxx"
)

add_library(unitree_ddsc SHARED IMPORTED GLOBAL)
set_target_properties(unitree_ddsc PROPERTIES
  IMPORTED_LOCATION "${UNITREE_DDS_C_LIBRARY}"
)

add_library(unitree_ddscxx SHARED IMPORTED GLOBAL)
set_target_properties(unitree_ddscxx PROPERTIES
  IMPORTED_LOCATION "${UNITREE_DDS_CXX_LIBRARY}"
)

add_library(unitree_g1d_driver_core src/driver_core.cpp)
target_include_directories(unitree_g1d_driver_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

add_executable(unitree_g1d_driver_node src/unitree_g1d_driver_node.cpp)
target_include_directories(unitree_g1d_driver_node PRIVATE include)
ament_target_dependencies(unitree_g1d_driver_node
  diagnostic_msgs
  geometry_msgs
  rclcpp
  std_msgs
)
target_link_libraries(unitree_g1d_driver_node
  unitree_g1d_driver_core
  unitree_sdk2_vendor
  unitree_ddscxx
  unitree_ddsc
  Threads::Threads
)
set_target_properties(unitree_g1d_driver_node PROPERTIES
  BUILD_RPATH "${UNITREE_DDS_LIBRARY_DIR}"
  INSTALL_RPATH "${UNITREE_DDS_LIBRARY_DIR}"
)

install(TARGETS
  unitree_g1d_driver_core
  unitree_g1d_driver_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY params/ DESTINATION share/${PROJECT_NAME}/params)
install(FILES README.md DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_driver_core test/test_driver_core.cpp)
  target_link_libraries(test_driver_core unitree_g1d_driver_core)
  target_include_directories(test_driver_core PRIVATE include)
endif()

ament_package()
