include(cc_library)
include(cc_test)

cc_library(
  NAME
   managers
  HDRS
    instance_mgr.h
    model_instance_mgr.h
    global_kvcache_mgr.h
  SRCS
    instance_mgr.cpp
    model_instance_mgr.cpp
    global_kvcache_mgr.cpp
  DEPS
    :chat_template
    :common
    :etcd_client
    :request
    :resource_model
    absl::random_random
    absl::strings
    glog::glog
    proto::proto_rpc_service
    proto::proto_disagg_pd_link
    proto_janus
)
target_link_libraries(managers PRIVATE brpc-static)
