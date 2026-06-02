enable_testing()

set(TEST_LAUNCHER "${CMAKE_CURRENT_LIST_DIR}/launcher.sh")

# Replay tests that do not restore captured host allocations at fixed addresses
# can run directly under sanitizers.
set(KKF_SANITIZER_COMPATIBLE_REPLAY_TESTS no_capture scalar_capture small_struct_functor)

# Fixed-address replay conflicts with sanitizer address-space reservations.
set(KKF_FIXED_HOST_ADDRESS_REPLAY_TESTS multiple_views_capture repeated_kernel_invocation struct_functor view_capture
                                        view_scalar_capture
)

function(add_capture_only_test TEST_NAME)
  set(KKF_CAPTURE_ONLY_TEST_NAME "capture_only_${TEST_NAME}")
  add_test(NAME "${KKF_CAPTURE_ONLY_TEST_NAME}"
           COMMAND "${TEST_LAUNCHER}" $<TARGET_FILE:kkf_kokkos_hooks> $<TARGET_FILE:${TEST_NAME}>
                   $<TARGET_FILE:${TEST_NAME}_replay> FALSE ${ARGN}
  )
  set_tests_properties(
    "${KKF_CAPTURE_ONLY_TEST_NAME}" PROPERTIES ENVIRONMENT KKF_CAPTURE_ONLY=1 LABELS
                                               "capture_only;sanitizer_compatible"
  )
endfunction()

function(add_replay_test TEST_NAME COMPARE_OUTPUT)
  add_test(NAME "${TEST_NAME}" COMMAND "${TEST_LAUNCHER}" $<TARGET_FILE:kkf_kokkos_hooks> $<TARGET_FILE:${TEST_NAME}>
                                       $<TARGET_FILE:${TEST_NAME}_replay> ${COMPARE_OUTPUT} ${ARGN}
  )

  if("${TEST_NAME}" IN_LIST KKF_SANITIZER_COMPATIBLE_REPLAY_TESTS)
    set(KKF_TEST_LABELS "replay;sanitizer_compatible")
  elseif("${TEST_NAME}" IN_LIST KKF_FIXED_HOST_ADDRESS_REPLAY_TESTS)
    set(KKF_TEST_LABELS "replay;fixed_host_address_replay")
    add_capture_only_test("${TEST_NAME}" ${ARGN})
  else()
    set(KKF_TEST_LABELS replay)
  endif()

  set_tests_properties("${TEST_NAME}" PROPERTIES LABELS "${KKF_TEST_LABELS}")
endfunction()

add_subdirectory(no_capture)
add_subdirectory(scalar_capture)
add_subdirectory(view_capture)
add_subdirectory(view_scalar_capture)
add_subdirectory(multiple_views_capture)
add_subdirectory(struct_functor)
add_subdirectory(small_struct_functor)
add_subdirectory(repeated_kernel_invocation)
