cmake_minimum_required(VERSION 3.23)

project(dlimgedit VERSION 1.0.0 LANGUAGES CXX)

option(DLIMGEDIT_TESTS "Build tests" ON)
option(DLIMGEDIT_INSTALL_MODELS "Install models" ON)
option(DLIMGEDIT_INSTALL_RUNTIME "Install runtime dependencies" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_RPATH $ORIGIN)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(depend/fmt)
add_subdirectory(depend/eigen)
add_subdirectory(depend/onnx)
add_subdirectory(depend/stb)
add_subdirectory(depend/dylib)
add_subdirectory(depend/catch2)

add_subdirectory(src)

add_subdirectory(models/segmentation)

if(DLIMGEDIT_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

install(TARGETS dlimgedit
  EXPORT dlimgeditTargets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
install(DIRECTORY src/include/ DESTINATION include)
install(FILES README.md LICENSE DESTINATION .)
if(DLIMGEDIT_INSTALL_MODELS)
  install(DIRECTORY models/ DESTINATION models FILES_MATCHING PATTERN "*.onnx")
endif()
if(DLIMGEDIT_INSTALL_RUNTIME)
  install(FILES ${ONNX_RUNTIME_DEPENDENCIES} DESTINATION bin)
endif()

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

if(WIN32)
  set(CPACK_GENERATOR "ZIP")
  set(CPACK_PACKAGE_FILE_NAME dlimgedit-windows-x64-${PROJECT_VERSION})
else()
  set(CPACK_GENERATOR "TGZ")
  set(CPACK_PACKAGE_FILE_NAME dlimgedit-linux-x64-${PROJECT_VERSION})
endif()
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
include(CPack)
