if(NOT TARGET dylib)
  include(FetchContent)

  FetchContent_Declare(
    dylib
    GIT_REPOSITORY https://github.com/martin-olivier/dylib
    GIT_TAG        v2.2.1
  )
  FetchContent_MakeAvailable(dylib)

endif() # NOT TARGET dylib