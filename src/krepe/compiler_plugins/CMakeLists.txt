if(NOT CMAKE_CXX_COMPILER_ID STREQUAL GNU)
  return()
endif()

execute_process(COMMAND gcc -print-file-name=plugin OUTPUT_VARIABLE GCC_PLUGIN_DIR_)
string(STRIP "${GCC_PLUGIN_DIR_}" GCC_PLUGIN_DIR)
if(GCC_PLUGIN_DIR STREQUAL "plugin")
  message(WARNING "This installation of GCC doesn't support plugins")
  return()
endif()

set(KREPE_HAS_GCC_PLUGIN TRUE PARENT_SCOPE)

add_library(krepe_gcc_plugin SHARED)
target_sources(krepe_gcc_plugin PRIVATE gcc.cpp)
target_include_directories(krepe_gcc_plugin PRIVATE "${GCC_PLUGIN_DIR}/include")
target_compile_options(krepe_gcc_plugin PRIVATE "-fno-rtti")
set_target_properties(krepe_gcc_plugin PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

install(TARGETS krepe_gcc_plugin)
