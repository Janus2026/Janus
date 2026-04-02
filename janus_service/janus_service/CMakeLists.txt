include(cc_binary)
include(cc_library)
include(cc_test)

add_subdirectory(proto)
add_subdirectory(common)
add_subdirectory(request)
add_subdirectory(rpc_service)
add_subdirectory(tokenizer)
add_subdirectory(chat_template)
add_subdirectory(http_service)
add_subdirectory(scheduler)

cc_binary(
  NAME
    janus_master_serving
  HDRS
    master.h
  SRCS
    master.cpp
  DEPS
    :janus_http_service
    :janus_rpc_service
)

