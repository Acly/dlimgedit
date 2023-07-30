add_executable(test_dlimgedit)
target_sources(test_dlimgedit PRIVATE
  test_image.cpp
  test_segmentation.cpp
  test_utils.cpp
)
target_include_directories(test_dlimgedit PRIVATE ../src)
target_link_libraries(test_dlimgedit PRIVATE
  dlimgedit fmt onnxruntime Eigen3::Eigen Catch2::Catch2WithMain
)

add_test(NAME test_dlimgedit COMMAND test_dlimgedit)
