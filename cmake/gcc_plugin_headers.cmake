if(KREPE_CXX_COMPILER_IS_NVCC)
  # NOTE: This should be modified if we ever need to call the host gcc with multiple flags
  set(GCC_EXE "${CMAKE_CXX_COMPILER};-c;-x;c++;/dev/null;-Xcompiler")
else()
  set(GCC_EXE "${CMAKE_CXX_COMPILER}")
endif()

execute_process(
  COMMAND ${GCC_EXE} -print-file-name=plugin OUTPUT_VARIABLE GCC_PLUGIN_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
)
if("${GCC_PLUGIN_DIR}" STREQUAL plugin OR NOT EXISTS "${GCC_PLUGIN_DIR}")
  message(WARNING "Your GCC installation does not support plugins")
  return()
endif()

if(EXISTS "${GCC_PLUGIN_DIR}/include/gcc-plugin.h")
  set(GCC_HAS_PLUGIN_HEADERS TRUE)
else()
  set(GCC_HAS_PLUGIN_HEADERS FALSE)
endif()

if(NOT GCC_HAS_PLUGIN_HEADERS AND NOT KREPE_BUILD_GCC_PLUGIN_HEADERS_IF_NEEDED)
  message(
    WARNING
      "Your GCC installation supports plugins, but the plugin headers cannot be found. You can install the necessary package or enable generating the missing plugin headers with -DKREPE_BUILD_GCC_PLUGIN_HEADERS_IF_NEEDED=ON"
  )
  return()
endif()

add_library(gcc-plugin-headers INTERFACE)

if(GCC_HAS_PLUGIN_HEADERS)
  message(STATUS "GCC installation supports plugins")
  target_include_directories(gcc-plugin-headers INTERFACE "${GCC_PLUGIN_DIR}/include")
  return()
endif()

message(
  STATUS "GCC installation supports plugins, but plugin headers are missing. Enabling generation of the missing headers"
)

include(ExternalProject)

execute_process(COMMAND ${GCC_EXE} -dumpfullversion OUTPUT_VARIABLE GCC_VERSION_ OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REGEX REPLACE [[[0-9]+$]] "0" GCC_VERSION "${GCC_VERSION_}")

execute_process(COMMAND ${GCC_EXE} -dumpmachine OUTPUT_VARIABLE GCC_TARGET OUTPUT_STRIP_TRAILING_WHITESPACE)

find_package(PkgConfig)

set(CONFIGURE_ARGS
    --enable-plugin
    --enable-languages=c,c++
    --disable-bootstrap
    --disable-multilib
    --disable-nls # we don't need translated diagnostics, allows to get rid of the gettext dependency
    --disable-lto
    --without-isl # we don't need polyhedral optimizations, allows to get rid of the isl dependency
    "--target=${GCC_TARGET}"
    "--host=${GCC_TARGET}"
    "--build=${GCC_TARGET}"
)
set(DEPS_TO_REMOVE "gettext;isl")
set(NO_DEPS TRUE)
macro(find_package_root PKG)
  find_package(${PKG} QUIET)

  if(NOT ${${PKG}_FOUND} OR NOT DEFINED ${PKG}_ROOT)
    if(PKG_CONFIG_FOUND)
      pkg_get_variable(${PKG}_ROOT ${PKG} "prefix")
    endif()
  endif()

  if(${PKG}_ROOT)
    list(APPEND CONFIGURE_ARGS "--with-${PKG}=${${PKG}_ROOT}")
    list(APPEND DEPS_TO_REMOVE ${PKG})
  else()
    message(WARNING "Failed to find ${PKG}, will be built alongside GCC")
    set(NO_DEPS FALSE)
  endif()
endmacro()

find_package_root(gmp)
find_package_root(mpfr)
find_package_root(mpc)

# FIXME: In order to not have two GCC source trees living on disk (the one we
# download and the copy made by ExternalProject_Add), we could only extract the
# `download_prerequisites` script, dowload the prerequisites and only keep the
# ones we need in the build dir. That way we could pass the archive to
# ExternalProject_Add and copy the prerequisites during the patch phase or
# before the build phase
set(GCC_SRC_DIR "${CMAKE_CURRENT_BINARY_DIR}/gcc-${GCC_VERSION}")
if(NOT EXISTS "${GCC_SRC_DIR}")
  message(STATUS "Downloading GCC ${GCC_VERSION}")
  set(GCC_ARCHIVE "${CMAKE_CURRENT_BINARY_DIR}/gcc-${GCC_VERSION}.tar.xz")
  file(DOWNLOAD https://ftp.gnu.org/gnu/gcc/gcc-${GCC_VERSION}/gcc-${GCC_VERSION}.tar.xz "${GCC_ARCHIVE}")
  message(STATUS "Extracting GCC ${GCC_VERSION}")
  file(ARCHIVE_EXTRACT INPUT "${GCC_ARCHIVE}" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

  if(NO_DEPS)
    message(STATUS "Found all GCC prerequisites, skipping download")
  else()
    message(STATUS "Dowloading GCC's prerequisites")
    execute_process(
      COMMAND ./contrib/download_prerequisites WORKING_DIRECTORY "${GCC_SRC_DIR}" OUTPUT_QUIET ERROR_QUIET
                                                                                  COMMAND_ERROR_IS_FATAL ANY
    )
    foreach(DEP ${DEPS_TO_REMOVE})
      file(GLOB FILES_TO_REMOVE "${GCC_SRC_DIR}/${DEP}*")
      file(REMOVE_RECURSE ${FILES_TO_REMOVE})
    endforeach()
  endif()
endif()

find_program(MAKE make REQUIRED)
find_program(SH sh REQUIRED)

ExternalProject_Add(
  gcc-plugin
  URL "${GCC_SRC_DIR}" DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  CONFIGURE_COMMAND ${SH} -c "cd <BINARY_DIR> && <SOURCE_DIR>/configure ${CONFIGURE_ARGS}"
  BUILD_COMMAND ${MAKE} configure-gcc
  COMMAND ${MAKE} all-libiberty all-libcpp all-libdecnumber all-libbacktrace
  COMMAND ${SH} -c "cd gcc && ${MAKE} install-plugin build_libsubdir= DESTDIR=<BINARY_DIR>/install"
  INSTALL_COMMAND ""
  LOG_CONFIGURE TRUE
  LOG_BUILD TRUE
  LOG_INSTALL TRUE
  LOG_OUTPUT_ON_FAILURE TRUE
)

ExternalProject_Add_Step(
  gcc-plugin find-headers
  COMMAND ${SH} -c "cp -r \$(dirname \$(dirname \$(find . -path '*/plugin/include/gcc-plugin.h' -type f))) <BINARY_DIR>"
  DEPENDEES build
  WORKING_DIRECTORY <BINARY_DIR>/install
  LOG TRUE
)
ExternalProject_Add_StepTargets(gcc-plugin find-headers)

target_link_libraries(gcc-plugin-headers INTERFACE gcc-plugin-find-headers)
ExternalProject_Get_Property(gcc-plugin BINARY_DIR)
target_include_directories(gcc-plugin-headers INTERFACE "${BINARY_DIR}/plugin/include")
