add_ops_compile_options(
    OP_NAME ReplaceToken
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Wno-error
)

target_sources(op_host_aclnn PRIVATE
replace_token_cpu.cpp
)

target_sources(optiling PRIVATE
    replace_token_cpu.cpp
)

target_include_directories(optiling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
    replace_token_cpu.cpp
)

install(FILES replace_token.cpp
    DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
