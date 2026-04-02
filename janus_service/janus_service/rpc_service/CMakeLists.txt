include(cc_binary)
include(cc_library)
include(cc_test)

cc_library(
  NAME
  janus_rpc_service
  HDRS
    service.h
  SRCS
    service.cpp
  DEPS
    :common
    :scheduler
    absl::random_random
    absl::strings
    glog::glog
    proto::proto_rpc_service
    proto_janus
    tokenizer
    chat_template
)
target_link_libraries(janus_rpc_service PRIVATE brpc-static)

cc_binary(
  NAME 
    janus_rpc_service_test
  SRCS 
    rpc_service_test.cpp
  DEPS
    :janus_rpc_service
    gflags::gflags
    glog::glog
    GTest::gtest_main
)
add_test(NAME JanusRpcServiceTest COMMAND janus_rpc_service_test)

cc_binary(
  NAME 
    janus_rpc_serving
  SRCS 
    main.cpp
  DEPS
    :janus_rpc_service
    gflags::gflags   
)

cc_library(
  NAME
  janus_rpc_client
  HDRS
    client.h
  SRCS
    client.cpp
  DEPS
    :common
    glog::glog
    proto::proto_rpc_service
)
target_link_libraries(janus_rpc_client PRIVATE brpc-static)
