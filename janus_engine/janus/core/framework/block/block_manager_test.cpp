#include <gtest/gtest.h>

#include "block_manager_impl.h"
#include "framework/xtensor/xtensor_kv_layout.h"

namespace janus {

TEST(BlockManagerTest, Basic) {
  const uint32_t n_blocks = 10;
  const uint32_t block_size = 2;
  BlockManager::Options options;
  options.num_blocks(n_blocks).block_size(block_size);
  BlockManagerImpl manager(options);

  EXPECT_EQ(manager.num_free_blocks(), n_blocks - 1);
  EXPECT_EQ(manager.block_size(), block_size);

  // Allocate a block
  {
    Block block = manager.allocate();
    EXPECT_EQ(block.id(), 1);
    EXPECT_EQ(block.size(), block_size);
    EXPECT_EQ(block.is_shared(), false);
    EXPECT_EQ(block.ref_count(), 1);

    EXPECT_EQ(manager.num_free_blocks(), n_blocks - 2);
  }
  // the block should be freed after the scope
  EXPECT_EQ(manager.num_free_blocks(), n_blocks - 1);

  // Allocate a list of blocks
  {
    std::vector<Block> blocks;
    for (uint32_t i = 1; i < n_blocks; ++i) {
      auto block = manager.allocate();
      EXPECT_EQ(block.id(), i);
      EXPECT_EQ(block.size(), block_size);
      EXPECT_EQ(block.is_shared(), false);
      EXPECT_EQ(block.ref_count(), 1);
      blocks.push_back(std::move(block));
    }
    EXPECT_EQ(manager.num_free_blocks(), 0);
    for (const auto& block : blocks) {
      EXPECT_EQ(block.ref_count(), 1);
      EXPECT_EQ(block.is_shared(), false);
    }

    // Test CHECK failure
    EXPECT_DEATH(manager.allocate(), "No more blocks available");
  }

  // all blocks should be freed after the scope
  EXPECT_EQ(manager.num_free_blocks(), n_blocks - 1);

  // Test shared blocks
  {
    Block block = manager.allocate();
    EXPECT_EQ(block.ref_count(), 1);
    EXPECT_EQ(block.is_shared(), false);
    // test copy constructor
    {
      // NOLINTNEXTLINE
      const Block block2 = block;
      EXPECT_EQ(block.ref_count(), 2);
      EXPECT_EQ(block.is_shared(), true);
      EXPECT_EQ(block2.ref_count(), 2);
      EXPECT_EQ(block2.is_shared(), true);
      EXPECT_EQ(block2, block);
    }
    EXPECT_EQ(block.ref_count(), 1);
    EXPECT_EQ(block.is_shared(), false);

    // test assignment operator
    {
      Block block4 = manager.allocate();
      block4 = block;
      EXPECT_EQ(block.ref_count(), 2);
      EXPECT_EQ(block.is_shared(), true);
      EXPECT_EQ(block4.ref_count(), 2);
      EXPECT_EQ(block4.is_shared(), true);
      EXPECT_EQ(block4, block);

      Block invalid_block;
      invalid_block = block;
      EXPECT_EQ(block.ref_count(), 3);
      EXPECT_EQ(block.is_shared(), true);
      EXPECT_EQ(invalid_block.ref_count(), 3);
      EXPECT_EQ(invalid_block.is_shared(), true);
      EXPECT_EQ(invalid_block, block);
    }
    EXPECT_EQ(block.ref_count(), 1);
    EXPECT_EQ(block.is_shared(), false);

    // test move constructor
    {
      Block block3 = std::move(block);
      EXPECT_FALSE(block.is_valid());

      EXPECT_EQ(block3.ref_count(), 1);
      EXPECT_EQ(block3.is_shared(), false);
      EXPECT_FALSE(block3 == block);
    }
    EXPECT_FALSE(block.is_valid());
  }
}

TEST(BlockManagerTest, XTensorLayoutKeepsLegacyStandardAttentionPacking) {
  constexpr uint64_t kPageSize = 2UL * 1024 * 1024;
  constexpr uint64_t kTensorBlockBytes = 16UL * 1024;

  const auto layout =
      build_xtensor_kv_layout(kPageSize, kTensorBlockBytes, kTensorBlockBytes);

  ASSERT_TRUE(layout.valid());
  EXPECT_EQ(layout.blocks_per_virt_page, 128);
  EXPECT_EQ(layout.k_pages_per_virt_page, 1);
  EXPECT_EQ(layout.v_pages_per_virt_page, 1);
  EXPECT_EQ(layout.index_pages_per_virt_page, 0);
}

TEST(BlockManagerTest, XTensorLayoutFindsDenserMlaPacking) {
  constexpr uint64_t kPageSize = 2UL * 1024 * 1024;
  constexpr uint64_t kBlockBytes = 16UL * 1024;
  constexpr uint64_t vBlockBytes = 4UL * 1024;

  const auto layout =
      build_xtensor_kv_layout(kPageSize, kBlockBytes, vBlockBytes);

  ASSERT_TRUE(layout.valid());
  EXPECT_EQ(layout.blocks_per_virt_page, 512);
  EXPECT_EQ(layout.k_pages_per_virt_page, 4);
  EXPECT_EQ(layout.v_pages_per_virt_page, 1);

  const uint64_t num_blocks = get_num_xtensor_blocks(
      /*num_phy_pages=*/1024, /*num_layers=*/2, layout);
  EXPECT_EQ(num_blocks, 52224);
}

}  // namespace janus
