add_ops_compile_options(
	OP_NAME XAttentionTl
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                # -g
                # -O0
                -I${CMAKE_SOURCE_DIR}/third_party/catlass/include
                -I${CMAKE_SOURCE_DIR}/common
                # -sanitizer
)

target_sources(op_host_aclnn PRIVATE
        x_attention_tl_cpu.cpp
)

# optiling
target_sources(optiling PRIVATE
        x_attention_tl_cpu.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)


# opproto
target_sources(opsproto PRIVATE
    x_attention_tl_cpu.cpp
)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# if(ASCEND_CUSTOM_COMPILE_OPTIONS)
#     set(ASCEND_CUSTOM_COMPILE_OPTIONS 
#         "${ASCEND_CUSTOM_COMPILE_OPTIONS} -I${CMAKE_CURRENT_SOURCE_DIR}")
# endif()
install(FILES x_attention_tl.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)


