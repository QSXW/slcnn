cmake_minimum_required(VERSION 3.20)

project(sl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SRC_FILES
    sl.h
)

set(PROJECT_FILES
    ${SRC_FILES}
)
source_group("\\" FILES ${SRC_FILES})
source_group("\\" FILES ${PROJECT_FILES})

add_library(${PROJECT_NAME} INTERFACE
    ${PROJECT_FILES})

target_include_directories(${PROJECT_NAME} INTERFACE ./)
