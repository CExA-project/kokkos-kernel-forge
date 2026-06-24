enable_testing()

set(TEST_LAUNCHER "${CMAKE_CURRENT_LIST_DIR}/launcher.sh")
add_custom_target(kkf_run_replay_tests)

function(kkf_set_test_working_directory TEST_NAME)
  set(TEST_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/work/${TEST_NAME}")
  file(MAKE_DIRECTORY "${TEST_WORKING_DIRECTORY}")
  set_tests_properties("${TEST_NAME}" PROPERTIES WORKING_DIRECTORY "${TEST_WORKING_DIRECTORY}")
endfunction()

function(add_replay_test TEST_NAME COMPARE_OUTPUT)
  set(TEST_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/work/${TEST_NAME}")

  add_test(NAME "${TEST_NAME}" COMMAND "${TEST_LAUNCHER}" $<TARGET_FILE:kkf_kokkos_hooks> $<TARGET_FILE:${TEST_NAME}>
                                       $<TARGET_FILE:${TEST_NAME}_replay> "${COMPARE_OUTPUT}" ${ARGN}
  )

  # Replay tests with fixed-address host allocations link to the unsanitized
  # replayer, while scalar replay tests use the sanitized replayer variant.
  set_tests_properties("${TEST_NAME}" PROPERTIES LABELS "replay;sanitizer_compatible")
  kkf_set_test_working_directory("${TEST_NAME}")

  add_custom_target(
    "kkf_run_${TEST_NAME}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${TEST_WORKING_DIRECTORY}"
    COMMAND "${CMAKE_COMMAND}" -E chdir "${TEST_WORKING_DIRECTORY}" "${TEST_LAUNCHER}" $<TARGET_FILE:kkf_kokkos_hooks>
            $<TARGET_FILE:${TEST_NAME}> $<TARGET_FILE:${TEST_NAME}_replay> "${COMPARE_OUTPUT}" ${ARGN}
    DEPENDS kkf_kokkos_hooks "${TEST_NAME}" "${TEST_NAME}_replay"
    USES_TERMINAL VERBATIM
  )
  add_dependencies(kkf_run_replay_tests "kkf_run_${TEST_NAME}")
endfunction()

add_subdirectory(no_capture)
add_subdirectory(scalar_capture)
add_subdirectory(view_capture)
add_subdirectory(view_scalar_capture)
add_subdirectory(multiple_views_capture)
add_subdirectory(struct_functor)
add_subdirectory(small_struct_functor)
add_subdirectory(repeated_kernel_invocation)
add_subdirectory(metadata)
