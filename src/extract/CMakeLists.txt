add_library(kernel_extractor)
target_sources(
  kernel_extractor PRIVATE kernel_extractor.cpp PUBLIC FILE_SET HEADERS BASE_DIRS . FILES kernel_extractor.hpp
)
target_link_libraries(
  kernel_extractor PRIVATE $<BUILD_LOCAL_INTERFACE:kkf_warnings> $<BUILD_LOCAL_INTERFACE:kkf_sanitizers>
)
target_link_libraries(kernel_extractor PUBLIC Kokkos::kokkos)
if(Kokkos_ENABLE_LIBDL)
  target_compile_definitions(kernel_extractor PRIVATE _GNU_SOURCE)
  target_link_libraries(kernel_extractor PRIVATE ${CMAKE_DL_LIBS})
endif()

install(TARGETS kernel_extractor EXPORT kkfTargets FILE_SET HEADERS)
