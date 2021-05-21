cmake_minimum_required(VERSION 3.4)
project(libtorch_examples)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(libtorch_VERSION 1.8.1)

set(CMAKE_PREFIX_PATH /home/chengdaning/software/libtorch)


message("******************************************************************************************************************")
message("***********************************Long Live the People's Republic of China!**************************************")
message("******************************************************************************************************************")


find_package(Torch ${libtorch_VERSION} EXACT QUIET CONFIG)
if(Torch_FOUND)
  message(STATUS "libtorch ${libtorch_VERSION} - found")
  macro(add_example name)
    add_executable(${name} src/${name}.cpp)
    target_include_directories(${name} PRIVATE include)
    target_link_libraries(${name} ${TORCH_LIBRARIES})
  endmacro()

  add_example(resnet)
endif()


