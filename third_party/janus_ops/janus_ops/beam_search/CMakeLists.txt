add_ops_compile_options(
	OP_NAME BeamSearch
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Wno-error
)

target_sources(op_host_aclnn PRIVATE
        beam_search_cpu.cpp
)

# optiling
target_sources(optiling PRIVATE
        beam_search_cpu.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)


# opproto
target_sources(opsproto PRIVATE
        beam_search_cpu.cpp
)

install(FILES beam_search.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES beam_search.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
