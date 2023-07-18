cmake_minimum_required(VERSION 3.24)
project(leetcode)

set(CMAKE_CXX_STANDARD 14)

include_directories(src)

add_executable(leetcode
        src/Solution.cpp src/SmallestInfiniteSet.cpp src/ATM.cpp src/MedianFinder.cpp src/Codec.cpp src/Solution2.cpp src/ParkingSystem.cpp src/MyHashSet.cpp src/UndergroundSystem.cpp src/SnapshotArray.cpp src/LRUCache.cpp)