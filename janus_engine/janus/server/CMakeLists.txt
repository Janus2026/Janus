include(cc_library)

cc_library(
  NAME
    janus_server
  HDRS
    janus_server.h
    janus_server_registry.h
  SRCS
    janus_server.cpp
    janus_server_registry.cpp
  DEPS
    :api_service
    :request
    absl::strings
    glog::glog
    proto::janus_proto
)
