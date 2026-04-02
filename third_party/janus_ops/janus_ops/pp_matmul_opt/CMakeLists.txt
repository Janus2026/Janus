add_ops_compile_options(
    OP_NAME PpMatmulOpt
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Wno-error
)

target_sources(op_host_aclnn PRIVATE
pp_matmul_opt_cpu.cpp
)

target_sources(optiling PRIVATE
    pp_matmul_opt_cpu.cpp
)

target_include_directories(optiling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
   pp_matmul_opt_cpu.cpp
)

install(FILES pp_matmul_opt.cpp
    DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
