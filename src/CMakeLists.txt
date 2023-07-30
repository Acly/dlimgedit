add_library(dlimgedit SHARED)

target_sources(dlimgedit PRIVATE
  dlimgedit.cpp
  environment.cpp
  image.cpp
  tensor.cpp
  segmentation.cpp
  session.cpp
)
target_compile_definitions(dlimgedit PRIVATE DLIMGEDIT_EXPORTS)
target_include_directories(dlimgedit PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE .)
target_link_libraries(dlimgedit PRIVATE fmt onnxruntime stb Eigen3::Eigen dylib)

if (MSVC)
  target_compile_options(dlimgedit PRIVATE /Zi /W4 /WX)
  target_compile_options(dlimgedit PRIVATE /wd4127 /wd5054) # for Eigen
  target_link_options(dlimgedit PRIVATE /DEBUG)

  foreach (file ${ONNX_RUNTIME_DEPENDENCIES})
    add_custom_command(TARGET dlimgedit POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${file} $<TARGET_FILE_DIR:dlimgedit>
    )
  endforeach()

  if (DLIMGEDIT_TESTS)
    set_target_properties(dlimgedit PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
  endif()
endif()
