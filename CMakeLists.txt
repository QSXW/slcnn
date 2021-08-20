cmake_minimum_required(VERSION 3.20)

project(slcnn LANGUAGES CXX)

set(CMAKE_VERBOSE_MAKEFILE True)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if (MSVC)
    add_compile_options(/Zc:alignedNew)
endif()

set(FRAMEWORK_FILES
    src/framework/Layer.cpp
    src/framework/Layer.h
    src/framework/Net.cpp
    src/framework/Net.h
    src/framework/Tensor.cpp
    src/framework/Tensor.h
    src/framework/Batch.cpp
    src/framework/Batch.h
    src/framework/Map.h
    src/framework/Helper.h
)

set(LAYER_FILES
    src/layer/BatchNormLayer.cpp
    src/layer/BatchNormLayer.h
    src/layer/ConvolutionalLayer.cpp
    src/layer/ConvolutionalLayer.h
    src/layer/ActivationLayer.cpp
    src/layer/ActivationLayer.h
    src/layer/MaxPoolLayer.cpp
    src/layer/MaxPoolLayer.h
    src/layer/SoftmaxLayer.cpp
    src/layer/SoftmaxLayer.h
    )

set(SRC_FILES
    src/slcnn.h
    src/main.cpp
    src/Test.cpp
    src/Test.h
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

add_subdirectory(sl)

target_link_libraries(${PROJECT_NAME}
    sl
)
