add_ops_compile_options(
    OP_NAME ConvertKvCacheFormat
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Werror
)

target_sources(op_host_aclnn PRIVATE
    convert_kv_cache_format_cpu.cpp
)

target_sources(optiling PRIVATE
    convert_kv_cache_format_cpu.cpp
)

target_include_directories(optiling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
    convert_kv_cache_format_cpu.cpp
)

install(FILES convert_kv_cache_format.cpp
    DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
