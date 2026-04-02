include(cc_test)

set(BINARY_DIR "${TRITON_BIN_PATH}")

cc_test(
  NAME
    rope_inplace_test
  SRCS
    triton_rope_inplace_test.cpp
  DEPS
    GTest::gtest
    GTest::gtest_main
    triton_adapter
    torch
    torch_npu
)

cc_test(
  NAME
    fused_gdn_gating_test
  SRCS
    triton_fused_gdn_gating_test.cpp
  DEPS
    GTest::gtest
    GTest::gtest_main
    triton_adapter
    torch
    torch_npu
)

cc_test(
  NAME
    fused_recurrent_gated_delta_rule_test
  SRCS
    triton_recurrent_gated_delta_rule_test.cpp
  DEPS
    GTest::gtest
    GTest::gtest_main
    triton_adapter
    torch
    torch_npu
)

cc_test(
  NAME
    layer_norm_fwd_test
  SRCS
    triton_layernorm_fwd_test.cpp
  DEPS
    GTest::gtest
    GTest::gtest_main
    triton_adapter
    torch
    torch_npu
)

cc_test(
  NAME
    causal_conv1d_update_test
  SRCS
    triton_causal_conv1d_update_test.cpp
  DEPS
    GTest::gtest
    GTest::gtest_main
    triton_adapter
    torch
    torch_npu
)


target_compile_definitions(fused_recurrent_gated_delta_rule_test PRIVATE
  TEST_BINARY_DIR="${BINARY_DIR}"
)


target_compile_definitions(fused_gdn_gating_test PRIVATE
  TEST_BINARY_DIR="${BINARY_DIR}"
)

target_compile_definitions(layer_norm_fwd_test PRIVATE
  TEST_BINARY_DIR="${BINARY_DIR}"
)

target_compile_definitions(causal_conv1d_update_test PRIVATE
  TEST_BINARY_DIR="${BINARY_DIR}"
)

target_compile_definitions(rope_inplace_test PRIVATE
  TEST_BINARY_DIR="${BINARY_DIR}"
)
