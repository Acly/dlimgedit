include(FetchContent)

FetchContent_Declare(
  onnxruntime
  URL https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-win-x64-gpu-1.15.1.zip
  DOWNLOAD_EXTRACT_TIMESTAMP true
)
FetchContent_MakeAvailable(onnxruntime)

add_library(onnxruntime_providers_shared SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime_providers_shared PROPERTIES
  IMPORTED_LOCATION ${onnxruntime_SOURCE_DIR}/lib/onnxruntime_providers_shared.dll
)
add_library(onnxruntime_providers_cuda SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime_providers_cuda PROPERTIES
  IMPORTED_LOCATION ${onnxruntime_SOURCE_DIR}/lib/onnxruntime_providers_cuda.dll
)
add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION ${onnxruntime_SOURCE_DIR}/lib/onnxruntime.dll
  IMPORTED_IMPLIB ${onnxruntime_SOURCE_DIR}/lib/onnxruntime.lib
  INTERFACE_INCLUDE_DIRECTORIES ${onnxruntime_SOURCE_DIR}/include
)
add_dependencies(onnxruntime onnxruntime_providers_shared onnxruntime_providers_cuda)
