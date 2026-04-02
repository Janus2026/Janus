include(grpc_proto_library)
include(proto_library)

proto_library(
  NAME
    proto_rpc_service
  SRCS
    janus_rpc_service.proto
)

proto_library(
  NAME
    proto_disagg_pd_link
  SRCS
    disagg_pd_link.proto
)

proto_library(
  NAME
    proto_http_service
  SRCS
    janus_http_service.proto
)

proto_library(
  NAME
    proto_janus
  SRCS
    janus/chat.proto
    janus/common.proto
    janus/completion.proto
)
