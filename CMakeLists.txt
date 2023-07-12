cmake_minimum_required(VERSION 3.23)

project(dlimgedit LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(depend/eigen)
add_subdirectory(depend/onnx)
add_subdirectory(depend/stb)
add_subdirectory(depend/catch2)

enable_testing()

add_library(dlimgedit)
target_sources(dlimgedit PRIVATE
  src/environment.cpp
  src/image.cpp
  src/tensor.cpp
  src/segmentation.cpp
)
target_include_directories(dlimgedit PUBLIC include PRIVATE src)
target_link_libraries(dlimgedit PRIVATE onnxruntime stb Eigen3::Eigen)
if (MSVC)
  target_compile_options(dlimgedit PRIVATE /W4 /WX)
  target_compile_options(dlimgedit PRIVATE /wd4127 /wd5054) # for Eigen
  add_custom_command(TARGET dlimgedit POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:onnxruntime> $<TARGET_FILE_DIR:dlimgedit>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE_DIR:onnxruntime>/onnxruntime_providers_shared.dll $<TARGET_FILE_DIR:dlimgedit>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE_DIR:onnxruntime>/onnxruntime_providers_cuda.dll $<TARGET_FILE_DIR:dlimgedit>)
endif()

add_executable(test_dlimgedit)
target_sources(test_dlimgedit PRIVATE
  test/test_image.cpp
  test/test_segmentation.cpp
  test/test_utils.cpp
)
target_include_directories(test_dlimgedit PRIVATE src test)
target_link_libraries(test_dlimgedit PRIVATE
  dlimgedit onnxruntime Eigen3::Eigen Catch2::Catch2WithMain
)
add_test(NAME test_dlimgedit COMMAND test_dlimgedit)
