if(NOT TARGET fmt)
  include(FetchContent)

  set(FMT_INSTALL OFF)
  FetchContent_Declare(fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        11.1.3
    GIT_SHALLOW    true
  )
  FetchContent_MakeAvailable(fmt)

endif() # fmt