cmake_minimum_required(VERSION 3.24)
project(leetcode)

set(CMAKE_CXX_STANDARD 14)

include_directories(src)

add_executable(leetcode
        src/Solution.cpp
        src/construct.cpp src/getFolderNames.cpp src/LinkedListRandomNode.cpp src/diameterOfBinaryTree.cpp src/MyLinkedList.cpp src/Trie.cpp)
