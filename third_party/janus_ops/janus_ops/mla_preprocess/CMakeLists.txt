add_ops_compile_options(
        OP_NAME MlaPreprocess
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Wno-error
)

target_sources(op_host_aclnn PRIVATE
        mla_preprocess_def.cpp
)

# optiling
target_sources(optiling PRIVATE
        mla_preprocess_tiling.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# opproto
target_sources(opsproto PRIVATE
        mla_preprocess_infershape.cpp
)

target_include_directories(opsproto PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)
install(FILES mla_preprocess.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES mla_preprocess_bf16.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES mla_preprocess_fp16.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES mla_preprocess_no_quant.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
