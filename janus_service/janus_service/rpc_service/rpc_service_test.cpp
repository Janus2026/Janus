#include <glog/logging.h>
#include <gtest/gtest.h>

#include "rpc_service/service.h"

namespace janus_service::test {

class JanusRpcServiceTest : public ::testing::Test {
 protected:
  void SetUp() override { google::InitGoogleLogging("JanusRpcServiceTest"); }

  void TearDown() override { google::ShutdownGoogleLogging(); }
};
// TODO
// TEST_F(JanusRpcServiceTest, RegisterInstance) {
//   RpcServiceConfig config;
//   HttpServiceConfig http_config;
//   ModelConfig model_config;
//   auto janus_service =
//       std::make_shared<JanusRpcServiceImpl>(config, model_config,
//       http_config);
//   std::string inst_name = "127.0.0.1@nic0";
//   InstanceMetaInfo metainfo(inst_name, "127.0.0.1:7777",
//   InstanceType::PREFILL); EXPECT_EQ(ErrorCode::OK,
//             janus_service->register_instance(inst_name, metainfo));

//   metainfo.type = InstanceType::DECODE;
//   EXPECT_EQ(ErrorCode::INSTANCE_EXISTED,
//             janus_service->register_instance(inst_name, metainfo));
// }

// TEST_F(JanusRpcServiceTest, UpdateInstanceMetainfo) {
//   RpcServiceConfig config;
//   HttpServiceConfig http_config;
//   ModelConfig model_config;
//   auto janus_service =
//       std::make_shared<JanusRpcServiceImpl>(config, model_config,
//       http_config);
//   std::string inst_name = "127.0.0.1@nic0";
//   InstanceMetaInfo metainfo(inst_name, "127.0.0.1:7777",
//   InstanceType::PREFILL); EXPECT_EQ(ErrorCode::OK,
//             janus_service->register_instance(inst_name, metainfo));
//   metainfo.type = InstanceType::DECODE;
//   EXPECT_EQ(ErrorCode::OK,
//             janus_service->update_instance_metainfo(inst_name, metainfo));

//   std::string inst_name2 = "127.0.0.1@nic2";
//   InstanceMetaInfo metainfo2(
//       inst_name2, "127.0.0.1:7778", InstanceType::PREFILL);
//   EXPECT_EQ(ErrorCode::INSTANCE_NOT_EXISTED,
//             janus_service->update_instance_metainfo(inst_name2, metainfo));
// }

}  // namespace janus_service::test
