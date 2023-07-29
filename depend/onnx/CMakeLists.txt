include(FetchContent)

if(WIN32)
  set(OS win)
  set(EXT zip)
else()
  set(OS linux)
  set(EXT tgz)
endif()
set(PREFIX ${CMAKE_SHARED_LIBRARY_PREFIX})
set(SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX})

FetchContent_Declare(
  onnxruntime
  URL https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-${OS}-x64-gpu-1.15.1.${EXT}
  DOWNLOAD_EXTRACT_TIMESTAMP true
)
FetchContent_MakeAvailable(onnxruntime)

add_library(onnxruntime_providers_shared SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime_providers_shared PROPERTIES
  IMPORTED_LOCATION ${onnxruntime_SOURCE_DIR}/lib/${PREFIX}onnxruntime_providers_shared${SUFFIX}
)
add_library(onnxruntime_providers_cuda SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime_providers_cuda PROPERTIES
  IMPORTED_LOCATION ${onnxruntime_SOURCE_DIR}/lib/${PREFIX}onnxruntime_providers_cuda${SUFFIX}
)
add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION ${onnxruntime_SOURCE_DIR}/lib/${PREFIX}onnxruntime${SUFFIX}
  IMPORTED_IMPLIB ${onnxruntime_SOURCE_DIR}/lib/onnxruntime.lib
  INTERFACE_INCLUDE_DIRECTORIES ${onnxruntime_SOURCE_DIR}/include
)
add_dependencies(onnxruntime onnxruntime_providers_shared onnxruntime_providers_cuda)

set(ONNX_RUNTIME_DEPENDENCIES
  $<TARGET_FILE:onnxruntime>
  $<TARGET_FILE:onnxruntime_providers_shared>
  $<TARGET_FILE:onnxruntime_providers_cuda>
  PARENT_SCOPE)
