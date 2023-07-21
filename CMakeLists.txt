cmake_minimum_required(VERSION 3.23)

project(dlimgedit LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(DLIMGEDIT_TESTS "Build tests" ON)

add_subdirectory(depend/eigen)
add_subdirectory(depend/onnx)
add_subdirectory(depend/stb)
add_subdirectory(depend/catch2)

add_library(dlimgedit SHARED)
target_sources(dlimgedit PRIVATE
  src/dlimgedit.cpp
  src/environment.cpp
  src/image.cpp
  src/tensor.cpp
  src/segmentation.cpp
)
target_compile_definitions(dlimgedit PRIVATE DLIMGEDIT_EXPORTS)
target_include_directories(dlimgedit PUBLIC include PRIVATE src)
target_link_libraries(dlimgedit PRIVATE onnxruntime stb Eigen3::Eigen)
set(DLIMGEDIT_RUNTIME_DEPENDENCIES
  $<TARGET_FILE:onnxruntime>
  $<TARGET_FILE:onnxruntime_providers_shared>
  $<TARGET_FILE:onnxruntime_providers_cuda>)
if (MSVC)
  target_compile_options(dlimgedit PRIVATE /W4 /WX)
  target_compile_options(dlimgedit PRIVATE /wd4127 /wd5054) # for Eigen
  foreach (file ${DLIMGEDIT_RUNTIME_DEPENDENCIES})
    add_custom_command(TARGET dlimgedit POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${file} $<TARGET_FILE_DIR:dlimgedit>
    )
  endforeach()
  if (DLIMGEDIT_TESTS)
    set_target_properties(dlimgedit PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
  endif()
endif()

install(TARGETS dlimgedit
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
install(FILES ${DLIMGEDIT_RUNTIME_DEPENDENCIES} DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY models/ DESTINATION models)

if (DLIMGEDIT_TESTS)
  enable_testing()

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
endif()
