#include "phy_page.h"

namespace janus {
PhyPage::PhyPage(torch::Device device, page_id_t page_id)
    : device_(device), page_id_(page_id) {
  int32_t device_id = device_.index();

  // create a physical memory handle for the device
  vmm::create_phy_mem_handle(phy_handle_, device_id);
}

PhyPage::~PhyPage() { vmm::release_phy_mem_handle(phy_handle_); }
}  // namespace janus