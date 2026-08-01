cmake_minimum_required(VERSION 3.10)
project(LeetCodeCSolutions)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_COMPILER "clang")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

include_directories(src)

if(APPLE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused")
endif()

file(GLOB C_SOURCES src/*.c)
foreach(C_SOURCE ${C_SOURCES})
    get_filename_component(SOURCE_NAME ${C_SOURCE} NAME_WE)
    add_executable(${SOURCE_NAME} ${C_SOURCE})
endforeach()