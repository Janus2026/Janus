add_ops_compile_options(
	OP_NAME BeamSearchGroup
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
        -Wno-error
                -I${CMAKE_SOURCE_DIR}/third_party/catlass/include
                -I${CMAKE_SOURCE_DIR}/common
)

target_sources(op_host_aclnn PRIVATE
        beam_search_group_cpu.cpp
)

# optiling
target_sources(optiling PRIVATE
        beam_search_group_cpu.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)


# opproto
target_sources(opsproto PRIVATE
        beam_search_group_cpu.cpp
)

install(FILES beam_search_group.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES beam_search_group.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
