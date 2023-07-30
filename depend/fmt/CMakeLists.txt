include(FetchContent)

set(FMT_INSTALL OFF)
FetchContent_Declare(fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG        10.0.0
  GIT_SHALLOW    true
)
FetchContent_MakeAvailable(fmt)
