if(NOT TARGET Eigen3::Eigen)
  include(FetchContent)
  
  FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    true
  )
  FetchContent_GetProperties(Eigen)
  if(NOT depname_POPULATED)
    FetchContent_Populate(Eigen)
    add_library(eigen INTERFACE)
    add_library(Eigen3::Eigen ALIAS eigen)
    target_include_directories(eigen INTERFACE ${eigen_SOURCE_DIR})
  endif()

endif() # Eigen3::Eigen