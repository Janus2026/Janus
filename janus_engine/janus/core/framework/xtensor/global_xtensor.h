#pragma once

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "phy_page.h"
#include "platform/vmm_api.h"

namespace janus {

/**
 * GlobalXTensor maps all physical pages into a single large XTensor-backed
 * virtual address space. It provides contiguous segment allocation for
 * model weights without per-page RPC mapping.
 *
 * This is a singleton (one per worker).
 */
class GlobalXTensor {
 public:
  // Get the global singleton instance
  static GlobalXTensor& get_instance() {
    static GlobalXTensor instance;
    return instance;
  }

  // Initialize (must be called after PhyPagePool::init)
  void init(const torch::Device& device);

  bool is_initialized() const { return initialized_; }

  // Get virtual address for a given page_id
  void* get_vaddr_by_page_id(page_id_t page_id) const;

  // Get base virtual address
  void* base_vaddr() const { return vir_ptr_to_void_ptr(vaddr_); }

  size_t total_size() const { return total_size_; }
  size_t num_total_pages() const { return num_total_pages_; }
  size_t page_size() const { return page_size_; }

  // Mooncake registration status (for idempotent registration)
  bool is_mooncake_registered() const { return mooncake_registered_; }
  void set_mooncake_registered(bool registered) {
    mooncake_registered_ = registered;
  }

 private:
  GlobalXTensor() = default;
  ~GlobalXTensor() = default;
  GlobalXTensor(const GlobalXTensor&) = delete;
  GlobalXTensor& operator=(const GlobalXTensor&) = delete;

  bool map_page(PhyPage* page, size_t offset);
  bool map_all_pages(const std::vector<PhyPage*>& pages);

  bool initialized_ = false;
  bool mooncake_registered_ = false;
  VirPtr vaddr_ = {};
  size_t total_size_ = 0;
  size_t page_size_ = 0;
  size_t num_total_pages_ = 0;
};

}  // namespace janus
