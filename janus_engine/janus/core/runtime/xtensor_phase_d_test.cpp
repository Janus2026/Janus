#include <gtest/gtest.h>

#include "distributed_runtime/engine.h"
#include "framework/batch/batch_forward_type.h"
#include "runtime/params_utils.h"

namespace janus {
namespace {

void populate_minimal_forward_input(proto::ForwardInput* pb_forward_input) {
  pb_forward_input->add_flatten_tokens_vec(1);
  pb_forward_input->add_flatten_positions_vec(0);
  pb_forward_input->set_batch_forward_type(BatchForwardType::PREFILL);
  pb_forward_input->set_max_seq_len(1);
  pb_forward_input->set_q_max_seq_len(1);
  pb_forward_input->add_seq_lens(1);
  pb_forward_input->add_q_seq_lens(1);
  pb_forward_input->add_kv_cache_tokens_nums(0);
  pb_forward_input->add_new_token_slot_ids(0);
  pb_forward_input->set_num_sequences(1);
  auto* block_table = pb_forward_input->add_block_tables_vec();
  block_table->add_block_tables(0);
}

TransferKVInfo build_transfer_kv_info(bool include_index) {
  TransferKVInfo info;
  info.request_id = include_index ? "req-new" : "req-legacy";
  info.local_blocks_ids = {1, 2};
  info.remote_blocks_ids = {10, 11};
  info.dp_rank = 0;
  info.remote_instance_info.name = "remote";
  info.remote_instance_info.rpc_address = "127.0.0.1:9000";
  info.remote_instance_info.type = "LLM";
  info.remote_instance_info.cluster_ids = {1234};
  info.remote_instance_info.addrs = {"mooncake://remote"};
  info.remote_instance_info.k_cache_ids = {20};
  info.remote_instance_info.v_cache_ids = {21};
  info.remote_instance_info.dp_size = 1;

  XTensorLayerOffsets layer0;
  layer0.k_offsets = {100, 120};
  layer0.v_offsets = {200, 220};
  if (include_index) {
    layer0.index_offsets = {300, 320};
  }

  XTensorLayerOffsets layer1;
  layer1.k_offsets = {400};
  layer1.v_offsets = {500};
  if (include_index) {
    layer1.index_offsets = {600};
  }

  info.dst_xtensor_layer_offsets = {layer0, layer1};
  if (include_index) {
    info.dst_xtensor_block_bytes.k_block_bytes = 16;
    info.dst_xtensor_block_bytes.v_block_bytes = 8;
    info.dst_xtensor_block_bytes.index_block_bytes = 4;
  }
  return info;
}

void expect_transfer_kv_info_eq(const TransferKVInfo& actual,
                                const TransferKVInfo& expected) {
  EXPECT_EQ(actual.request_id, expected.request_id);
  EXPECT_EQ(actual.local_blocks_ids, expected.local_blocks_ids);
  EXPECT_EQ(actual.remote_blocks_ids, expected.remote_blocks_ids);
  EXPECT_EQ(actual.dp_rank, expected.dp_rank);
  EXPECT_EQ(actual.remote_instance_info.name,
            expected.remote_instance_info.name);
  EXPECT_EQ(actual.remote_instance_info.rpc_address,
            expected.remote_instance_info.rpc_address);
  EXPECT_EQ(actual.remote_instance_info.type,
            expected.remote_instance_info.type);
  EXPECT_EQ(actual.remote_instance_info.cluster_ids,
            expected.remote_instance_info.cluster_ids);
  EXPECT_EQ(actual.remote_instance_info.addrs,
            expected.remote_instance_info.addrs);
  EXPECT_EQ(actual.remote_instance_info.k_cache_ids,
            expected.remote_instance_info.k_cache_ids);
  EXPECT_EQ(actual.remote_instance_info.v_cache_ids,
            expected.remote_instance_info.v_cache_ids);
  EXPECT_EQ(actual.remote_instance_info.dp_size,
            expected.remote_instance_info.dp_size);
  ASSERT_EQ(actual.dst_xtensor_layer_offsets.size(),
            expected.dst_xtensor_layer_offsets.size());
  for (size_t i = 0; i < actual.dst_xtensor_layer_offsets.size(); ++i) {
    EXPECT_EQ(actual.dst_xtensor_layer_offsets[i].k_offsets,
              expected.dst_xtensor_layer_offsets[i].k_offsets);
    EXPECT_EQ(actual.dst_xtensor_layer_offsets[i].v_offsets,
              expected.dst_xtensor_layer_offsets[i].v_offsets);
    EXPECT_EQ(actual.dst_xtensor_layer_offsets[i].index_offsets,
              expected.dst_xtensor_layer_offsets[i].index_offsets);
  }
  EXPECT_EQ(actual.dst_xtensor_block_bytes.k_block_bytes,
            expected.dst_xtensor_block_bytes.k_block_bytes);
  EXPECT_EQ(actual.dst_xtensor_block_bytes.v_block_bytes,
            expected.dst_xtensor_block_bytes.v_block_bytes);
  EXPECT_EQ(actual.dst_xtensor_block_bytes.index_block_bytes,
            expected.dst_xtensor_block_bytes.index_block_bytes);
}

}  // namespace

TEST(ParamsUtilsTest, ProtoToForwardInputKeepsLegacyXTensorTransferInfo) {
  proto::ForwardInput pb_forward_input;
  populate_minimal_forward_input(&pb_forward_input);

  const TransferKVInfo expected =
      build_transfer_kv_info(/*include_index=*/false);
  auto* pb_transfer = pb_forward_input.add_transfer_kv_infos();
  pb_transfer->set_request_id(expected.request_id);
  pb_transfer->add_local_blocks_ids(expected.local_blocks_ids[0]);
  pb_transfer->add_local_blocks_ids(expected.local_blocks_ids[1]);
  pb_transfer->add_remote_blocks_ids(expected.remote_blocks_ids[0]);
  pb_transfer->add_remote_blocks_ids(expected.remote_blocks_ids[1]);
  pb_transfer->set_dp_rank(expected.dp_rank);
  pb_transfer->mutable_remote_instance_info()->set_name(
      expected.remote_instance_info.name);
  pb_transfer->mutable_remote_instance_info()->set_rpc_address(
      expected.remote_instance_info.rpc_address);
  pb_transfer->mutable_remote_instance_info()->set_type(
      expected.remote_instance_info.type);
  pb_transfer->mutable_remote_instance_info()->add_cluster_ids(
      expected.remote_instance_info.cluster_ids[0]);
  pb_transfer->mutable_remote_instance_info()->add_addrs(
      expected.remote_instance_info.addrs[0]);
  pb_transfer->mutable_remote_instance_info()->add_k_cache_ids(
      expected.remote_instance_info.k_cache_ids[0]);
  pb_transfer->mutable_remote_instance_info()->add_v_cache_ids(
      expected.remote_instance_info.v_cache_ids[0]);
  pb_transfer->mutable_remote_instance_info()->set_dp_size(
      expected.remote_instance_info.dp_size);
  for (const auto& layer_offsets : expected.dst_xtensor_layer_offsets) {
    auto* pb_layer_offsets = pb_transfer->add_dst_xtensor_layer_offsets();
    for (uint64_t offset : layer_offsets.k_offsets) {
      pb_layer_offsets->add_k_offsets(offset);
    }
    for (uint64_t offset : layer_offsets.v_offsets) {
      pb_layer_offsets->add_v_offsets(offset);
    }
  }

  ForwardInput forward_input;
  proto_to_forward_input(&pb_forward_input, forward_input, 1);

  ASSERT_EQ(forward_input.transfer_kv_infos.size(), 1);
  const auto& actual = forward_input.transfer_kv_infos.front();
  expect_transfer_kv_info_eq(actual, expected);
  EXPECT_FALSE(actual.dst_xtensor_block_bytes.valid());
}

TEST(ParamsUtilsTest,
     ForwardInputProtoRoundTripKeepsIndexOffsetsAndBlockBytes) {
  RawForwardInput raw_forward_input;
  raw_forward_input.flatten_tokens_vec = {1};
  raw_forward_input.flatten_positions_vec = {0};
  raw_forward_input.batch_forward_type = BatchForwardType::PREFILL;
  raw_forward_input.max_seq_len = 1;
  raw_forward_input.q_max_seq_len = 1;
  raw_forward_input.seq_lens = {1};
  raw_forward_input.q_seq_lens = {1};
  raw_forward_input.kv_cache_tokens_nums = {0};
  raw_forward_input.new_token_slot_ids = {0};
  raw_forward_input.block_tables_vec = {{0}};
  raw_forward_input.num_sequences = 1;
  raw_forward_input.transfer_kv_infos = {
      build_transfer_kv_info(/*include_index=*/true)};

  proto::ForwardInput pb_forward_input;
  forward_input_to_proto(raw_forward_input, &pb_forward_input);

  ForwardInput forward_input;
  proto_to_forward_input(&pb_forward_input, forward_input, 1);

  ASSERT_EQ(forward_input.transfer_kv_infos.size(), 1);
  expect_transfer_kv_info_eq(forward_input.transfer_kv_infos.front(),
                             raw_forward_input.transfer_kv_infos.front());
  EXPECT_TRUE(
      forward_input.transfer_kv_infos.front().dst_xtensor_block_bytes.valid());
  EXPECT_TRUE(forward_input.transfer_kv_infos.front()
                  .dst_xtensor_block_bytes.has_index());
}

TEST(KVCacheCapacityUtilsTest, SharedBlockCountUsesExactXTensorPagePacking) {
  constexpr uint64_t kPageSize = 100;

  Engine::KVCacheCapacity target_kv_cache_cap;
  target_kv_cache_cap.n_blocks = 64;
  target_kv_cache_cap.n_layers = 1;
  target_kv_cache_cap.slot_size = 2;
  target_kv_cache_cap.xtensor_kv_layout.page_size = kPageSize;
  target_kv_cache_cap.xtensor_kv_layout.k_block_bytes = 1;
  target_kv_cache_cap.xtensor_kv_layout.v_block_bytes = 1;
  target_kv_cache_cap.xtensor_kv_layout.blocks_per_virt_page = 3;
  target_kv_cache_cap.xtensor_kv_layout.k_pages_per_virt_page = 1;
  target_kv_cache_cap.xtensor_kv_layout.v_pages_per_virt_page = 1;

  Engine::KVCacheCapacity draft_kv_cache_cap;
  draft_kv_cache_cap.n_blocks = 64;
  draft_kv_cache_cap.n_layers = 1;
  draft_kv_cache_cap.slot_size = 3;
  draft_kv_cache_cap.xtensor_kv_layout.page_size = kPageSize;
  draft_kv_cache_cap.xtensor_kv_layout.k_block_bytes = 1;
  draft_kv_cache_cap.xtensor_kv_layout.v_block_bytes = 1;
  draft_kv_cache_cap.xtensor_kv_layout.blocks_per_virt_page = 4;
  draft_kv_cache_cap.xtensor_kv_layout.k_pages_per_virt_page = 2;
  draft_kv_cache_cap.xtensor_kv_layout.v_pages_per_virt_page = 1;
  EXPECT_EQ(calculate_shared_kv_cache_blocks(/*cache_size_in_bytes=*/700,
                                             target_kv_cache_cap,
                                             draft_kv_cache_cap,
                                             /*block_size=*/1,
                                             kPageSize),
            4);
}

TEST(KVCacheCapacityUtilsTest, SharedBlockCountIncludesAuxiliarySlotSizes) {
  Engine::KVCacheCapacity target_kv_cache_cap;
  target_kv_cache_cap.n_blocks = 128;
  target_kv_cache_cap.n_layers = 2;
  target_kv_cache_cap.slot_size = 8;
  target_kv_cache_cap.index_slot_size = 4;
  target_kv_cache_cap.extra_slot_size = 2;

  Engine::KVCacheCapacity draft_kv_cache_cap;
  draft_kv_cache_cap.n_blocks = 128;
  draft_kv_cache_cap.n_layers = 1;
  draft_kv_cache_cap.slot_size = 6;
  draft_kv_cache_cap.index_slot_size = 1;
  draft_kv_cache_cap.extra_slot_size = 3;

  constexpr int32_t kBlockSize = 16;
  const int64_t bytes_per_shared_block =
      static_cast<int64_t>(get_kv_cache_required_bytes(target_kv_cache_cap,
                                                       kBlockSize,
                                                       /*n_blocks=*/1)) +
      static_cast<int64_t>(get_kv_cache_required_bytes(draft_kv_cache_cap,
                                                       kBlockSize,
                                                       /*n_blocks=*/1));

  EXPECT_EQ(calculate_shared_kv_cache_blocks(
                /*cache_size_in_bytes=*/bytes_per_shared_block * 7 + 100,
                target_kv_cache_cap,
                draft_kv_cache_cap,
                kBlockSize),
            7);
}

}  // namespace janus
