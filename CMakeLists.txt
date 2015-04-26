macro(required)
  foreach(name ${ARGN})
    find_package(${name} REQUIRED)
  endforeach()
endmacro()

macro(executables)
  foreach(driver ${ARGN})
    ADD_EXECUTABLE(${driver} ${driver}.cpp)
    TARGET_LINK_LIBRARIES(${driver}  ${libs})
  endforeach()
endmacro()

macro(cuda_executables)
  foreach(driver ${ARGN})
    CUDA_COMPILE(${driver}_O ${driver}.cu)
    CUDA_ADD_EXECUTABLE(${driver} ${${driver}_O})
    TARGET_LINK_LIBRARIES(${driver}  ${libs})
  endforeach()
endmacro()

cmake_minimum_required(VERSION 2.8)

required(CUDA MPI)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_35,code=sm_35)

set (CMAKE_CXX_FLAGS "-O3 -Wall -DSPLATT_LOG_LEVEL=10")

find_path (splatter_include splatter.h
"../splatt/include" )

find_library(splatter_lib splatter 
"../splatt/src/build" )

message( ${splatter_include} -:- ${splatter_lib})

set(libs ${splatter_lib} ${MPI_CXX_LIBRARIES} ${CUDA_LIBRARIES})

include_directories (${splatter_include} ${MPI_CXX_INCLUDE_PATH})

cuda_compile(TETS_O tet_vols.cu)
cuda_add_executable(tet_vols ${TETS_O} splatt_cuda.cpp)
TARGET_LINK_LIBRARIES(tet_vols  ${libs})

cuda_compile(saxpy_o saxpy.cu)
cuda_add_executable(saxpy ${saxpy_o})

cuda_compile(MEM_O mem_check.cu)
cuda_add_executable(cuda_mem ${MEM_O} mem_driver.cpp)
target_link_libraries(cuda_mem ${libs})
