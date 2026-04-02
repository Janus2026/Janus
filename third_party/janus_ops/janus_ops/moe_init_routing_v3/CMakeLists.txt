add_ops_compile_options(
        OP_NAME MoeInitRoutingV3
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Wno-error
)
target_sources(op_host_aclnnExc PRIVATE
        moe_init_routing_v3_def.cpp
)

# optiling
target_sources(opapi PRIVATE
        moe_init_routing_v3_l0.cpp
        aclnn_moe_init_routing_v3.cpp
)

target_sources(optiling PRIVATE
        moe_init_routing_v3_tiling_base.cpp
        moe_init_routing_v3_tiling.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/common/
)

# opproto
target_sources(opsproto PRIVATE
        moe_init_routing_v3_infershape.cpp
)

target_include_directories(opsproto PRIVATE
        ${CMAKE_SOURCE_DIR}/common/
)

install(FILES moe_init_routing_v3.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES aclnn_moe_init_routing_v3.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)
