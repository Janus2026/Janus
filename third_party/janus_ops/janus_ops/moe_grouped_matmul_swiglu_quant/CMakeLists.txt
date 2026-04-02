add_ops_compile_options(
        OP_NAME MoeGroupedMatmulSwigluQuant
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Wno-error
)

target_sources(op_host_aclnnExc PRIVATE
        moe_grouped_matmul_swiglu_quant_def.cpp
)

target_sources(opapi PRIVATE
        moe_grouped_matmul_swiglu_quant_l0.cpp
        aclnn_moe_grouped_matmul_swiglu_quant.cpp
)

target_sources(optiling PRIVATE
        moe_grouped_matmul_swiglu_quant_tiling.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
        moe_grouped_matmul_swiglu_quant_infershape.cpp
)

install(FILES moe_grouped_matmul_swiglu_quant.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES moe_grouped_matmul_swiglu_quant.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES aclnn_moe_grouped_matmul_swiglu_quant.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)
