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

include(parmetis.txt)

cmake_minimum_required(VERSION 2.8)

required(MPI CUDA)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20)

set (CMAKE_CXX_FLAGS "-O3 -Wall -DSPLATT_LOG_LEVEL=10")

include_directories ("../broke_splatt/include" ${MPI_CXX_INCLUDE_PATH} ${PARMETIS_INCLUDE_DIR})

find_library(SPLATTER splatter ../broke_splatt/src)

set(libs ${MPI_CXX_LIBRARIES} ${PARMETIS_LIBRARIES} ${SPLATTER}) # duma efence

cuda_compile(TETS_O tet_vols.cu)

cuda_add_executable(tet_vols ${TETS_O} splatt_cuda.cpp)
TARGET_LINK_LIBRARIES(tet_vols  ${libs})

cuda_compile(MEM_O mem_check.cu)
cuda_add_executable(cuda_mem ${MEM_O} mem_driver.cpp)
target_link_libraries(cuda_mem ${libs})
