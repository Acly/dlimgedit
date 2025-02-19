add_executable(test_dlimgedit)
target_sources(test_dlimgedit PRIVATE
  test_ggml.cpp
  test_ggml_mobile.cpp
  test_image.cpp
  test_segmentation.cpp
  test_utils.cpp
)
target_include_directories(test_dlimgedit PRIVATE ../src)
target_link_libraries(test_dlimgedit PRIVATE
  dlimgedit fmt onnxruntime Eigen3::Eigen ggml Catch2::Catch2WithMain
)

if (MSVC)
  target_compile_options(test_dlimgedit PRIVATE /Zi /W4) # /WX
  target_compile_options(test_dlimgedit PRIVATE /wd4127 /wd5054 /wd4245) # for Eigen
  target_link_options(test_dlimgedit PRIVATE /DEBUG)
endif()

add_test(NAME test_dlimgedit COMMAND test_dlimgedit)


add_library(dlimgedit_workbench workbench.cpp)
target_include_directories(dlimgedit_workbench PRIVATE ../src)
target_link_libraries(dlimgedit_workbench PRIVATE 
  dlimgedit fmt onnxruntime ggml
)
if (MSVC)
  target_compile_options(dlimgedit_workbench PRIVATE /Zi /Od)
  target_link_options(dlimgedit_workbench PRIVATE /DEBUG)
endif()