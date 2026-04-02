#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace janus {

struct XTensorKvPageLayout {
  uint64_t page_size = 0;
  uint64_t k_block_bytes = 0;
  uint64_t v_block_bytes = 0;
  uint64_t index_block_bytes = 0;
  uint32_t blocks_per_virt_page = 0;
  uint32_t k_pages_per_virt_page = 0;
  uint32_t v_pages_per_virt_page = 0;
  uint32_t index_pages_per_virt_page = 0;

  [[nodiscard]] bool has_index() const noexcept {
    return index_block_bytes > 0 && index_pages_per_virt_page > 0;
  }

  [[nodiscard]] bool valid() const noexcept {
    return page_size > 0 && k_block_bytes > 0 && v_block_bytes > 0 &&
           blocks_per_virt_page > 0 && k_pages_per_virt_page > 0 &&
           v_pages_per_virt_page > 0;
  }

  [[nodiscard]] uint32_t total_tensor_pages_per_virt_page() const noexcept {
    return k_pages_per_virt_page + v_pages_per_virt_page +
           index_pages_per_virt_page;
  }
};

inline uint64_t ceil_div_u64(uint64_t value, uint64_t divisor) {
  return (value + divisor - 1) / divisor;
}

inline uint64_t page_period_for_block_bytes(uint64_t page_size,
                                            uint64_t block_bytes) {
  if (page_size == 0 || block_bytes == 0) {
    return 0;
  }
  return page_size / std::gcd(page_size, block_bytes);
}

inline XTensorKvPageLayout build_xtensor_kv_layout(
    uint64_t page_size,
    uint64_t k_block_bytes,
    uint64_t v_block_bytes,
    uint64_t index_block_bytes = 0,
    uint32_t max_blocks_per_virt_page = 4096) {
  XTensorKvPageLayout layout{};
  if (page_size == 0 || k_block_bytes == 0 || v_block_bytes == 0) {
    return layout;
  }

  std::vector<uint64_t> tensor_block_bytes = {k_block_bytes, v_block_bytes};
  if (index_block_bytes > 0) {
    tensor_block_bytes.push_back(index_block_bytes);
  }

  uint64_t search_limit = 1;
  for (uint64_t bytes : tensor_block_bytes) {
    search_limit =
        std::max(search_limit, page_period_for_block_bytes(page_size, bytes));
  }
  search_limit = std::min<uint64_t>(search_limit, max_blocks_per_virt_page);

  uint64_t best_blocks_per_virt_page = 0;
  uint64_t best_total_pages = std::numeric_limits<uint64_t>::max();
  uint64_t best_k_pages = 0;
  uint64_t best_v_pages = 0;
  uint64_t best_index_pages = 0;

  for (uint64_t blocks_per_virt_page = 1; blocks_per_virt_page <= search_limit;
       ++blocks_per_virt_page) {
    const uint64_t k_pages =
        ceil_div_u64(blocks_per_virt_page * k_block_bytes, page_size);
    const uint64_t v_pages =
        ceil_div_u64(blocks_per_virt_page * v_block_bytes, page_size);
    const uint64_t index_pages =
        index_block_bytes == 0
            ? 0
            : ceil_div_u64(blocks_per_virt_page * index_block_bytes, page_size);
    const uint64_t total_pages = k_pages + v_pages + index_pages;
    if (total_pages == 0) {
      continue;
    }

    const bool better_density = best_blocks_per_virt_page == 0 ||
                                blocks_per_virt_page * best_total_pages >
                                    best_blocks_per_virt_page * total_pages;
    const bool same_density_smaller_blocks =
        best_blocks_per_virt_page != 0 &&
        blocks_per_virt_page * best_total_pages ==
            best_blocks_per_virt_page * total_pages &&
        blocks_per_virt_page < best_blocks_per_virt_page;
    if (!better_density && !same_density_smaller_blocks) {
      continue;
    }

    best_blocks_per_virt_page = blocks_per_virt_page;
    best_total_pages = total_pages;
    best_k_pages = k_pages;
    best_v_pages = v_pages;
    best_index_pages = index_pages;
  }

  if (best_blocks_per_virt_page == 0) {
    return layout;
  }

  layout.page_size = page_size;
  layout.k_block_bytes = k_block_bytes;
  layout.v_block_bytes = v_block_bytes;
  layout.index_block_bytes = index_block_bytes;
  layout.blocks_per_virt_page =
      static_cast<uint32_t>(best_blocks_per_virt_page);
  layout.k_pages_per_virt_page = static_cast<uint32_t>(best_k_pages);
  layout.v_pages_per_virt_page = static_cast<uint32_t>(best_v_pages);
  layout.index_pages_per_virt_page = static_cast<uint32_t>(best_index_pages);
  return layout;
}

inline uint64_t get_num_xtensor_virt_pages(uint64_t num_phy_pages,
                                           int64_t num_layers,
                                           const XTensorKvPageLayout& layout) {
  if (num_layers <= 0 || !layout.valid()) {
    return 0;
  }
  const uint64_t phy_pages_per_virt_page =
      static_cast<uint64_t>(num_layers) *
      layout.total_tensor_pages_per_virt_page();
  if (phy_pages_per_virt_page == 0) {
    return 0;
  }
  return num_phy_pages / phy_pages_per_virt_page;
}

inline uint64_t get_num_xtensor_blocks(uint64_t num_phy_pages,
                                       int64_t num_layers,
                                       const XTensorKvPageLayout& layout) {
  return get_num_xtensor_virt_pages(num_phy_pages, num_layers, layout) *
         layout.blocks_per_virt_page;
}

inline uint64_t get_tensor_offset_for_block(uint64_t block_id,
                                            uint64_t tensor_block_bytes,
                                            uint32_t tensor_pages_per_virt_page,
                                            const XTensorKvPageLayout& layout) {
  if (!layout.valid() || tensor_block_bytes == 0 ||
      tensor_pages_per_virt_page == 0) {
    return 0;
  }
  const uint64_t virt_page_id = block_id / layout.blocks_per_virt_page;
  const uint64_t in_page_block_id = block_id % layout.blocks_per_virt_page;
  return virt_page_id * tensor_pages_per_virt_page * layout.page_size +
         in_page_block_id * tensor_block_bytes;
}

}  // namespace janus
