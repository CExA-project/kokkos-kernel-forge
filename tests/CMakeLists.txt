enable_testing()

set(TEST_LAUNCHER "${CMAKE_CURRENT_LIST_DIR}/launcher.sh")

function(add_replay_test TEST_NAME COMPARE_OUTPUT)
  add_test(NAME "${TEST_NAME}" COMMAND "${TEST_LAUNCHER}" $<TARGET_FILE:kkf_kokkos_hooks> $<TARGET_FILE:${TEST_NAME}>
                                       $<TARGET_FILE:${TEST_NAME}_replay> ${COMPARE_OUTPUT} ${ARGN}
  )
endfunction()

add_subdirectory(no_capture)
add_subdirectory(scalar_capture)
add_subdirectory(view_capture)
add_subdirectory(view_scalar_capture)
add_subdirectory(multiple_views_capture)
add_subdirectory(struct_functor)
add_subdirectory(small_struct_functor)
add_subdirectory(repeated_kernel_invocation)
