include(FetchContent)

FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG        5736b15f7ea0ffb08dd38af21067c314d6a3aae9
)
FetchContent_MakeAvailable(stb)

add_library(stb stb.cpp)
target_include_directories(stb PUBLIC ${stb_SOURCE_DIR})
