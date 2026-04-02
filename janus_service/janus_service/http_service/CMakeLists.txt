include(cc_binary)
include(cc_library)

cc_library(
  NAME
  janus_http_service
  HDRS
    service.h
    request_tracer.h
  SRCS
    service.cpp 
    request_tracer.cpp
  DEPS
    :common
    :request
    :scheduler
    absl::random_random
    absl::synchronization
    glog::glog
    nlohmann_json::nlohmann_json
    proto::proto_http_service
    proto_janus
)
target_link_libraries(janus_http_service PRIVATE brpc-static leveldb::leveldb ZLIB::ZLIB protobuf::libprotobuf OpenSSL::SSL OpenSSL::Crypto)

cc_binary(
  NAME 
    janus_http_serving
  SRCS 
    main.cpp
  DEPS
    :janus_http_service
    gflags::gflags
)
