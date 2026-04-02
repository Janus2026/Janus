#include "virt_page.h"

#include <glog/logging.h>

#include <algorithm>
#include <stdexcept>

#include "common/macros.h"

namespace janus {

VirtPage::VirtPage(int64_t page_id, size_t page_size)
    : page_id_(page_id), page_size_(page_size) {}

void VirtPage::require_init() const {
  CHECK(start_block_.has_value()) << "VirtPage not initialised";
  CHECK(end_block_.has_value()) << "VirtPage not initialised";
  CHECK(num_kv_blocks_.has_value()) << "VirtPage not initialised";
}

void VirtPage::init(size_t block_mem_size) {
  auto [start, end] = get_block_range(page_id_, page_size_, block_mem_size);
  start_block_ = start;
  end_block_ = end;
  num_kv_blocks_ = end - start;

  free_list_.clear();
  free_list_.reserve(*num_kv_blocks_);
  for (int64_t i = start; i < end; ++i) {
    free_list_.push_back(i);
  }
}

void VirtPage::init_fixed_blocks(size_t blocks_per_page) {
  CHECK_GT(blocks_per_page, 0) << "blocks_per_page must be > 0";

  start_block_ = page_id_ * static_cast<int64_t>(blocks_per_page);
  end_block_ = *start_block_ + static_cast<int64_t>(blocks_per_page);
  num_kv_blocks_ = blocks_per_page;

  free_list_.clear();
  free_list_.reserve(*num_kv_blocks_);
  for (int64_t i = *start_block_; i < *end_block_; ++i) {
    free_list_.push_back(i);
  }
}

std::vector<int64_t> VirtPage::alloc(size_t num_blocks) {
  require_init();
  if (full()) {
    throw std::runtime_error("VirtPage " + std::to_string(page_id_) +
                             " is already full");
  }

  size_t actual_alloc = std::min(num_blocks, free_list_.size());
  std::vector<int64_t> block_ids(free_list_.begin(),
                                 free_list_.begin() + actual_alloc);
  free_list_.erase(free_list_.begin(), free_list_.begin() + actual_alloc);
  return block_ids;
}

void VirtPage::free(int64_t block_id) {
  require_init();
  free_list_.push_back(block_id);
}

void VirtPage::free_batch(const std::vector<int64_t>& block_ids) {
  require_init();
  free_list_.insert(free_list_.end(), block_ids.begin(), block_ids.end());
}

bool VirtPage::empty() const {
  require_init();
  return free_list_.size() == *num_kv_blocks_;
}

bool VirtPage::full() const {
  require_init();
  return free_list_.empty();
}

size_t VirtPage::num_free_blocks() const {
  require_init();
  return free_list_.size();
}

const std::vector<int64_t>& VirtPage::get_free_blocks() const {
  require_init();
  return free_list_;
}

std::pair<int64_t, int64_t> VirtPage::get_block_range(int64_t page_id,
                                                      size_t page_size,
                                                      size_t block_mem_size) {
  int64_t start_block =
      (page_id * page_size + block_mem_size - 1) / block_mem_size;
  int64_t end_block = ((page_id + 1) * page_size) / block_mem_size;
  return {start_block, end_block};
}

size_t VirtPage::get_num_blocks(size_t page_size, size_t block_mem_size) {
  return page_size / block_mem_size;
}

}  // namespace janus
