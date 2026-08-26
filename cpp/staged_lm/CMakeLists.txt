cmake_minimum_required(VERSION 3.16)
project(staged_lm LANGUAGES CXX)

option(STAGED_LM_EIGEN_VECTORIZE "Enable Eigen packet vectorization" ON)
option(STAGED_LM_MARCH_NATIVE "Compile for the build host ISA" ON)
option(STAGED_LM_ENABLE_CUDA "Build the CUDA value-forward evaluator" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(staged_lm STATIC
    src/binary_io.cpp
    src/cpu_evaluators.cpp
    src/cuda_evaluators.cpp
    src/densekp.cpp
    src/densekp_output.cpp
    src/erp.cpp
    src/evaluator.cpp
    src/lm.cpp
    src/multiframe.cpp
    src/pipeline.cpp
    src/pose_forward.cpp
    src/pose_jacobian.cpp
    src/pose_kernels.cpp
    src/pose_lm.cpp
    src/pose_normal_equations.cpp
    src/pose_shape_forward.cpp
    src/pose_shape_jacobian.cpp
    src/pose_shape_lm.cpp
    src/pose_shape_normal_equations.cpp
    src/pose_shape_schur.cpp
    src/shape.cpp
    src/so3.cpp
    src/stage1.cpp
    src/stage1_lm.cpp
)
include(CheckLanguage)
if(STAGED_LM_ENABLE_CUDA)
  check_language(CUDA)
endif()
if(STAGED_LM_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  add_library(densekp_erp_remap_cuda SHARED src/densekp_erp_remap_cuda.cu)
  set_target_properties(densekp_erp_remap_cuda PROPERTIES
      CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  target_sources(staged_lm PRIVATE src/cuda_forward.cu)
  find_path(STAGED_LM_NVTX_INCLUDE_DIR nvtx3/nvToolsExt.h
      HINTS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} /usr/local/cuda/include)
  find_library(STAGED_LM_NVTX_LIBRARY nvToolsExt
      HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} /usr/local/cuda/lib64)
  if(STAGED_LM_NVTX_INCLUDE_DIR AND STAGED_LM_NVTX_LIBRARY)
    target_include_directories(staged_lm PRIVATE ${STAGED_LM_NVTX_INCLUDE_DIR})
    target_link_libraries(staged_lm PUBLIC ${STAGED_LM_NVTX_LIBRARY})
    target_compile_definitions(staged_lm PRIVATE STAGED_LM_HAS_NVTX=1)
  else()
    target_compile_definitions(staged_lm PRIVATE STAGED_LM_HAS_NVTX=0)
  endif()
  set_target_properties(staged_lm PROPERTIES
      CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  target_compile_definitions(staged_lm PUBLIC STAGED_LM_HAS_CUDA=1)
  target_compile_options(staged_lm PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--prec-div=true;--prec-sqrt=true>)
else()
  target_sources(staged_lm PRIVATE src/cuda_forward_stub.cpp)
  target_compile_definitions(staged_lm PUBLIC STAGED_LM_HAS_CUDA=0)
  target_compile_definitions(staged_lm PRIVATE STAGED_LM_HAS_NVTX=0)
endif()
target_include_directories(staged_lm PUBLIC include)
target_link_libraries(staged_lm PUBLIC Eigen3::Eigen)
target_compile_options(staged_lm PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-ffp-contract=off>)
if(NOT STAGED_LM_EIGEN_VECTORIZE)
  target_compile_definitions(staged_lm PUBLIC EIGEN_DONT_VECTORIZE)
endif()
if(STAGED_LM_MARCH_NATIVE)
  target_compile_options(staged_lm PRIVATE -march=native)
endif()

function(staged_lm_app target source)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE staged_lm)
  target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
  if(STAGED_LM_MARCH_NATIVE)
    target_compile_options(${target} PRIVATE -march=native)
  endif()
endfunction()

staged_lm_app(stage1_reference_main apps/stage1_reference_main.cpp)
staged_lm_app(pose_reference_main apps/pose_reference_main.cpp)
staged_lm_app(pose_lm_reference_main apps/pose_lm_reference_main.cpp)
staged_lm_app(pose_shape_reference_main apps/pose_shape_reference_main.cpp)
staged_lm_app(pose_shape_jacobian_reference_main apps/pose_shape_jacobian_reference_main.cpp)
staged_lm_app(pose_shape_ne_reference_main apps/pose_shape_ne_reference_main.cpp)
staged_lm_app(pose_shape_schur_reference_main apps/pose_shape_schur_reference_main.cpp)
staged_lm_app(pose_shape_lm_reference_main apps/pose_shape_lm_reference_main.cpp)
staged_lm_app(staged_lm_pipeline_reference_main apps/staged_lm_pipeline_reference_main.cpp)
staged_lm_app(multiframe_staged_lm_reference_main apps/multiframe_staged_lm_reference_main.cpp)

foreach(target staged_lm_pipeline_reference_main multiframe_staged_lm_reference_main)
  target_compile_definitions(${target} PRIVATE
      STAGED_LM_EXPECT_EIGEN_VECTORIZE=$<BOOL:${STAGED_LM_EIGEN_VECTORIZE}>
      STAGED_LM_EXPECT_MARCH_NATIVE=$<BOOL:${STAGED_LM_MARCH_NATIVE}>)
endforeach()

enable_testing()
function(staged_lm_test target source)
  staged_lm_app(${target} ${source})
  add_test(NAME ${target} COMMAND ${target})
endfunction()

staged_lm_test(stage1_cpp_self_test tests/stage1_cpp_self_test.cpp)
staged_lm_test(pose_cpp_self_test tests/pose_cpp_self_test.cpp)
staged_lm_test(pose_lm_cpp_self_test tests/pose_lm_cpp_self_test.cpp)
staged_lm_test(pose_shape_cpp_self_test tests/pose_shape_cpp_self_test.cpp)
staged_lm_test(pose_shape_jacobian_cpp_self_test tests/pose_shape_jacobian_cpp_self_test.cpp)
staged_lm_test(pose_shape_ne_cpp_self_test tests/pose_shape_ne_cpp_self_test.cpp)
staged_lm_test(pose_shape_schur_cpp_self_test tests/pose_shape_schur_cpp_self_test.cpp)
staged_lm_test(pose_shape_lm_cpp_self_test tests/pose_shape_lm_cpp_self_test.cpp)
staged_lm_test(staged_lm_pipeline_cpp_self_test tests/staged_lm_pipeline_cpp_self_test.cpp)
staged_lm_test(multiframe_staged_lm_cpp_self_test tests/multiframe_staged_lm_cpp_self_test.cpp)
staged_lm_test(densekp_cpp_self_test tests/densekp_cpp_self_test.cpp)
