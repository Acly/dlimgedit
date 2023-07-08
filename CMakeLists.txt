cmake_minimum_required(VERSION 3.23)

project(dlimgedit LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

include(FetchContent)

FetchContent_Declare(
  onnxruntime
  URL https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-win-x64-1.15.1.zip
  DOWNLOAD_EXTRACT_TIMESTAMP true
)
FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG        5736b15f7ea0ffb08dd38af21067c314d6a3aae9
)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.3.2
)
FetchContent_MakeAvailable(onnxruntime stb Catch2)

add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION ${onnxruntime_SOURCE_DIR}/lib/onnxruntime.dll
  IMPORTED_IMPLIB ${onnxruntime_SOURCE_DIR}/lib/onnxruntime.lib
  INTERFACE_INCLUDE_DIRECTORIES ${onnxruntime_SOURCE_DIR}/include
)

add_library(stb)
target_sources(stb PRIVATE src/stb.cpp)
target_include_directories(stb PUBLIC ${stb_SOURCE_DIR})

add_library(dlimgedit)
target_sources(dlimgedit PRIVATE
  src/image.cpp
  src/onnx.cpp
  src/segmentation.cpp
)
target_include_directories(dlimgedit PUBLIC include PRIVATE src)
target_link_libraries(dlimgedit PRIVATE onnxruntime stb)

add_executable(test_dlimgedit)
target_sources(test_dlimgedit PRIVATE
  test/test_image.cpp
  test/test_segmentation.cpp
)
target_link_libraries(test_dlimgedit PRIVATE dlimgedit Catch2::Catch2WithMain)
add_test(NAME test_dlimgedit COMMAND test_dlimgedit)
