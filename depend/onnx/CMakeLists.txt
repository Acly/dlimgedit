include(FetchContent)

if(WIN32)
  # Use onnxruntime with DirectML

  # DirectML.dll is installed with windows, but the version is typically too old.
  FetchContent_Declare(
      directml
      DOWNLOAD_EXTRACT_TIMESTAMP true
      URL https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/1.12.0
  )
  FetchContent_Declare(
    onnxruntime
    DOWNLOAD_EXTRACT_TIMESTAMP true
    URL https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/Microsoft.ML.OnnxRuntime.DirectML.1.15.1.zip
  )
  FetchContent_MakeAvailable(directml onnxruntime)

  add_library(directml SHARED IMPORTED GLOBAL)
  set_target_properties(directml PROPERTIES
    IMPORTED_LOCATION ${directml_SOURCE_DIR}/bin/x64-win/DirectML.dll
    IMPORTED_IMPLIB   ${directml_SOURCE_DIR}/bin/x64-win/DirectML.lib
  )
  add_library(onnxruntime SHARED IMPORTED GLOBAL)
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION ${onnxruntime_SOURCE_DIR}/runtimes/win-x64/native/onnxruntime.dll
    IMPORTED_IMPLIB ${onnxruntime_SOURCE_DIR}/runtimes/win-x64/native/onnxruntime.lib
    INTERFACE_INCLUDE_DIRECTORIES ${onnxruntime_SOURCE_DIR}/build/native/include
  )
  add_dependencies(onnxruntime directml)
  set(ONNX_RUNTIME_DEPENDENCIES
    $<TARGET_FILE:onnxruntime>
    $<TARGET_FILE:directml>
    PARENT_SCOPE)

else() # Linux
  # Use onnxruntime with CUDA

  FetchContent_Declare(
    onnxruntime
    DOWNLOAD_EXTRACT_TIMESTAMP true
    URL https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-linux-x64-gpu-1.15.1.tgz
  )
  FetchContent_MakeAvailable(onnxruntime)

  add_library(onnxruntime_providers_shared SHARED IMPORTED GLOBAL)
  set_target_properties(onnxruntime_providers_shared PROPERTIES
    IMPORTED_LOCATION ${onnxruntime_SOURCE_DIR}/lib/libonnxruntime_providers_shared.so
  )
  add_library(onnxruntime_providers_cuda SHARED IMPORTED GLOBAL)
  set_target_properties(onnxruntime_providers_cuda PROPERTIES
    IMPORTED_LOCATION ${onnxruntime_SOURCE_DIR}/lib/libonnxruntime_providers_cuda.so
  )
  add_library(onnxruntime SHARED IMPORTED GLOBAL)
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION ${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so
    INTERFACE_INCLUDE_DIRECTORIES ${onnxruntime_SOURCE_DIR}/include
  )
  add_dependencies(onnxruntime onnxruntime_providers_shared onnxruntime_providers_cuda)

endif()
