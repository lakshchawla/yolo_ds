cmake_minimum_required(VERSION 3.20) # 3.20+ is highly recommended for modern CUDA support

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.8/bin/nvcc")

set(PROJECT_N yolo_ds)
project(${PROJECT_N} VERSION 1.0 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_ARCHITECTURES "native") # Optimizes CUDA kernels for your specific GPU

# Catch the standard DeepStream C files along with your C++ and CUDA files
file(GLOB_RECURSE SRCS
        ${PROJECT_SOURCE_DIR}/src/*.cpp
        ${PROJECT_SOURCE_DIR}/src/*.c
        ${PROJECT_SOURCE_DIR}/src/*.cu
)

# Macro that gets all of the header containing directories.
MACRO(header_directories return_list includes_base_folder extention)
    FILE(GLOB_RECURSE new_list ${includes_base_folder}/*.${extention})
    SET(dir_list "")
    FOREACH(file_path ${new_list})
        GET_FILENAME_COMPONENT(dir_path ${file_path} PATH)
        SET(dir_list ${dir_list} ${dir_path})
    ENDFOREACH()
    LIST(REMOVE_DUPLICATES dir_list)
    SET(${return_list} ${dir_list})
ENDMACRO()

header_directories(INCLUDES ${PROJECT_SOURCE_DIR}/include/ hpp)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0)

find_package(CUDAToolkit REQUIRED)

set(NVDS_VERSION "8.0")
set(DS_DIR "/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}")
set(DS_INCLUDE_DIR "${DS_DIR}/sources/includes")
set(DS_LIB_DIR "${DS_DIR}/lib")

# Find DeepStream libraries
find_library(NVDSGST_META_LIB NAMES nvdsgst_meta PATHS ${DS_LIB_DIR} REQUIRED)
find_library(NVDS_META_LIB NAMES nvds_meta PATHS ${DS_LIB_DIR} REQUIRED)
find_library(NVDS_YML_PARSER_LIB NAMES nvds_yml_parser PATHS ${DS_LIB_DIR} REQUIRED)
find_library(NVDSGST_HELPER_LIB NAMES nvdsgst_helper PATHS ${DS_LIB_DIR} REQUIRED) # Fixed typo here

add_executable(${PROJECT_N}
        ${SRCS}
        test/transit_utils.h
)

target_compile_definitions(${PROJECT_N} PRIVATE
        CONFIG_DIR="${PROJECT_SOURCE_DIR}/include"
)

target_include_directories(${PROJECT_N} PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${GST_INCLUDE_DIRS}
        ${DS_INCLUDE_DIR}
        # CUDAToolkit automatically attaches its own includes
)

target_link_libraries(${PROJECT_N} PUBLIC
        ${GST_LIBRARIES}
        ${NVDSGST_META_LIB}
        ${NVDS_META_LIB}
        ${NVDS_YML_PARSER_LIB}
        ${NVDSGST_HELPER_LIB}
        CUDA::cudart # Translates -lcudart
        CUDA::cuda_driver   # Translates -lcuda
        m            # Standard math library (Translates -lm)
)

set_target_properties(${PROJECT_N} PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "${DS_LIB_DIR}"
)

message("src files:")
foreach(file ${SRCS})
    message(STATUS ${file})
endforeach()

message("include directories:")
foreach(dir ${INCLUDES})
    message(STATUS ${dir})
endforeach()