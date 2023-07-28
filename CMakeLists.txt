cmake_minimum_required(VERSION 3.23)

project(dlimgedit VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(DLIMGEDIT_TESTS "Build tests" ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(depend/eigen)
add_subdirectory(depend/onnx)
add_subdirectory(depend/stb)
add_subdirectory(depend/catch2)

add_subdirectory(src)

if (DLIMGEDIT_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

install(TARGETS dlimgedit
  EXPORT dlimgeditTargets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
install(FILES ${ONNX_RUNTIME_DEPENDENCIES} DESTINATION bin)
install(DIRECTORY src/include/ DESTINATION include)
install(DIRECTORY models/ DESTINATION models)
install(FILES README.md LICENSE DESTINATION .)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  dlimgeditConfigVersion.cmake
  VERSION ${PACKAGE_VERSION}
  COMPATIBILITY AnyNewerVersion
)
install(EXPORT dlimgeditTargets
  FILE dlimgeditConfig.cmake
  CONFIGURATIONS Release
  DESTINATION lib/cmake/dlimgedit
)
