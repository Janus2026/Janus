add_ops_compile_options(
    OP_NAME CacheUnsharedKV
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Wno-error
)

target_sources(op_host_aclnn PRIVATE
cache_unshared_kv_cpu.cpp
)

target_sources(optiling PRIVATE
    cache_unshared_kv_cpu.cpp
)

target_include_directories(optiling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
    cache_unshared_kv_cpu.cpp
)

install(FILES cache_unshared_kv.cpp
    DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
