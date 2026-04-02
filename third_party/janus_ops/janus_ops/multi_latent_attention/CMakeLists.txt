add_ops_compile_options(
	OP_NAME MultiLatentAttention
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -DASCENDC_DUMP=0
)

target_sources(op_host_aclnn PRIVATE
        multi_latent_attention_cpu.cpp
)

# optiling
target_sources(optiling PRIVATE
        multi_latent_attention_cpu.cpp
        math.cpp
        multi_latent_attention_tiling_impl.cpp
        multi_latent_attention_tiling_dependency.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)


# opproto
target_sources(opsproto PRIVATE
        multi_latent_attention_cpu.cpp
)

install(FILES multi_latent_attention.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)




