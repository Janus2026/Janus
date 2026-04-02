add_ops_compile_options(
        OP_NAME MlaPreprocessV2
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Wno-error
)

target_sources(op_host_aclnn PRIVATE
        mla_preprocess_v2_def.cpp
)

# optiling
target_sources(optiling PRIVATE
        mla_preprocess_v2_tiling.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# opproto
target_sources(opsproto PRIVATE
        mla_preprocess_v2_infershape.cpp
)

target_include_directories(opsproto PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)
install(FILES mla_preprocess_v2.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
