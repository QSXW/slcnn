cmake_minimum_required(VERSION 3.20)

project(slcnn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(FRAMEWORK_FILES
    src/framework/Layer.cpp
    src/framework/Layer.h
    src/framework/Net.cpp
    src/framework/Net.h
)

set(LAYER_FILES
    src/layer/BatchNormLayer.cpp
    src/layer/BatchNormLayer.h
    src/layer/ConvolutionLayer.cpp
    src/layer/ConvolutionLayer.h
    src/layer/ReLuLayer.cpp
    src/layer/ReLuLayer.h
    )

set(SRC_FILES
    src/slcnn.h
    src/main.cpp
)

set(PROJECT_FILES
    ${FRAMEWORK_FILES}
    ${SRC_FILES}
    ${LAYER_FILES}
)

source_group("\\framework\\" FILES ${FRAMEWORK_FILES})
source_group("\\" FILES ${SRC_FILES})
source_group("\\" FILES ${PROJECT_FILES})

add_executable(${PROJECT_NAME}
    ${PROJECT_FILES})

target_include_directories(${PROJECT_NAME} PRIVATE src/)
