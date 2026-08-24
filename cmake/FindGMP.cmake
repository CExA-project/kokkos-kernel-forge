# This script is a slightly modified version of https://github.com/navcoin/navcoin-core/blob/baf2287239a1b8f69445c959e28737276f9d68a6/src/mcl/cmake/FindGMP.cmake

# FindGMP.cmake
#
# Finds the GNU Multiple Precision Arithmetic Library (GMP)
# See http://gmplib.org/
#
# This will define the following variables::
#
#    GMP_FOUND
#    GMP_VERSION
#    GMP_DEFINITIONS
#    GMP_INCLUDE_DIR
#    GMP_LIBRARY
#
# and the following imported targets::
#
#     GMP::GMP

find_package(PkgConfig QUIET)
pkg_check_modules(PC_GMP QUIET gmp)

set(GMP_VERSION ${PC_GMP_VERSION})

find_library(GMP_LIBRARY NAMES gmp libgmp HINTS ENV GMP_ROOT ${GMP_ROOT} ${PC_GMP_LIBDIR} ${PC_GMP_LIBRARY_DIRS})

find_path(GMP_INCLUDE_DIR NAMES gmp.h HINTS ENV GMP_ROOT ${GMP_ROOT} ${PC_GMP_INCLUDEDIR} ${PC_GMP_INCLUDE_DIRS})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GMP REQUIRED_VARS GMP_INCLUDE_DIR GMP_LIBRARY VERSION_VAR GMP_VERSION)

if(GMP_FOUND)
  set(GMP_LIBRARIES ${GMP_LIBRARY})
  set(GMP_INCLUDE_DIRS ${GMP_INCLUDE_DIR})
  set(GMP_DEFINITIONS ${PC_GMP_CFLAGS_OTHER})

  if(NOT TARGET GMP::GMP)
    add_library(GMP::GMP UNKNOWN IMPORTED)
    set_target_properties(
      GMP::GMP
      PROPERTIES INTERFACE_COMPILE_OPTIONS "${PC_GMP_CFLAGS_OTHER}" INTERFACE_INCLUDE_DIRECTORIES "${GMP_INCLUDE_DIR}"
                 IMPORTED_LOCATION "${GMP_LIBRARY}"
    )
  endif()
endif()

mark_as_advanced(GMP_FOUND GMP_INCLUDE_DIR GMP_LIBRARY)
