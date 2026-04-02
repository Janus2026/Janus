#include "mapping_npu.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

namespace janus {

MappingNPU::Options get_mapping_options() {
  MappingNPU::Options options;
  options.dp_size(2)
      .tp_size(8)
      .moe_tp_size(2)
      .moe_ep_size(8)
      .pp_size(1)
      .sp_size(1);
  return options;
}

TEST(TestMappingNPU, ToJson) {
  std::string rank_table_file;
  MappingNPU::Options options = get_mapping_options();
  MappingNPU mapping(rank_table_file, 16, 6, options);
  nlohmann::json data = mapping.to_json();
  LOG(INFO) << "Mapping INFO:\n" << data.dump(2);
  nlohmann::json attn_dp = data["attnDp"];
  int32_t attn_dp_group_id = attn_dp["groupId"];
  EXPECT_EQ(attn_dp_group_id, 6);
  nlohmann::json attn_tp = data["attnTp"];
  int32_t attn_tp_group_id = attn_tp["groupId"];
  EXPECT_EQ(attn_tp_group_id, 0);
  nlohmann::json mlp_tp = data["mlpTp"];
  int32_t mlp_tp_buffer_size = mlp_tp["bufferSize"];
  EXPECT_EQ(mlp_tp_buffer_size, 128);
}

}  // namespace janus