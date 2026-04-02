#pragma once

#include <torch/torch.h>

#include "platform/vmm_api.h"

namespace janus {

// Page ID type for physical page identification
using page_id_t = int64_t;

class PhyPage {
 public:
  // Constructor with page_id (-1 means unassigned)
  PhyPage(torch::Device device, page_id_t page_id = -1);

  ~PhyPage();

  const torch::Device& device() const { return device_; }

  PhyMemHandle get_phy_handle() const { return phy_handle_; }

  // Get the page ID
  page_id_t page_id() const { return page_id_; }

 private:
  torch::Device device_;
  PhyMemHandle phy_handle_;
  page_id_t page_id_;  // Unique identifier for this page in the pool
};
}  // namespace janus