add_library(krepe_plugin_dep INTERFACE)
install(TARGETS krepe_plugin_dep EXPORT krepeTargets)

if(NOT KREPE_ENABLE_COMPILER_PLUGINS)
  return()
endif()

set(KREPE_USES_NVCC_WRAPPER FALSE)
if(Kokkos_ENABLE_CUDA)
  kokkos_cxx_compiler_cuda_test(KREPE_CXX_COMPILER_BUILDS_CUDA "${CMAKE_CXX_COMPILER}")
  kokkos_compiler_is_nvcc(KREPE_CXX_COMPILER_IS_NVCC "${CMAKE_CXX_COMPILER}")
  if(NOT KREPE_CXX_COMPILER_BUILDS_CUDA OR KREPE_CXX_COMPILER_IS_NVCC)
    set(KREPE_USES_NVCC_WRAPPER TRUE)
  endif()
  unset(KREPE_CXX_COMPILER_BUILDS_CUDA)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
  include("${PROJECT_SOURCE_DIR}/cmake/gcc_plugin_headers.cmake")

  if(NOT TARGET gcc-plugin-headers)
    return()
  endif()

  add_library(krepe_gcc_plugin SHARED gcc.cpp)
  target_link_libraries(krepe_gcc_plugin PRIVATE gcc-plugin-headers)
  target_compile_options(krepe_gcc_plugin PRIVATE "-fno-rtti")
  set_target_properties(krepe_gcc_plugin PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

  # Since We cannot create a direct depency to the plugin using
  # target_link_libraries, we crate a custom command depending on the plugin
  # which creates a header that gets included in kernel_extractor.hpp. This
  # results in a dependency chain like this:
  #   exe -> extractor -> dummy header -> plugin
  # The header will be regenerated if the plugin needs to be rebuilt, which will
  # lead to a recompilation of the executables
  set(DUMMY_PLUGIN_HEADER "${CMAKE_BINARY_DIR}/dummy.h")
  add_custom_command(
    OUTPUT "${DUMMY_PLUGIN_HEADER}" COMMAND cmake -E touch "${DUMMY_PLUGIN_HEADER}" DEPENDS krepe_gcc_plugin
  )
  add_custom_target(dummy "" DEPENDS "${DUMMY_PLUGIN_HEADER}")
  if(KREPE_USES_NVCC_WRAPPER)
    target_compile_options(
      krepe_plugin_dep
      INTERFACE $<BUILD_INTERFACE:-Xcompiler;-fplugin=$<TARGET_FILE:krepe_gcc_plugin>;-include;${DUMMY_PLUGIN_HEADER}>
                $<INSTALL_INTERFACE:-Xcompiler;-fplugin=${CMAKE_INSTALL_FULL_LIBDIR}/libkrepe_gcc_plugin.so>
    )
  else()
    target_compile_options(
      krepe_plugin_dep
      INTERFACE $<BUILD_INTERFACE:-fplugin=$<TARGET_FILE:krepe_gcc_plugin>;-include;${DUMMY_PLUGIN_HEADER}>
                $<INSTALL_INTERFACE:-fplugin=${CMAKE_INSTALL_FULL_LIBDIR}/libkrepe_gcc_plugin.so>
    )
  endif()
  add_dependencies(krepe_plugin_dep dummy)

  install(TARGETS krepe_gcc_plugin)
endif()
