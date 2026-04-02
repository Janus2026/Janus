#include "xtensor_dist_service.h"

#include <brpc/closure_guard.h>
#include <brpc/controller.h>
#include <glog/logging.h>

#include <vector>

#include "common/device_monitor.h"
#include "global_xtensor.h"
#include "phy_page_pool.h"
#include "platform/device.h"
#include "xtensor_allocator.h"

namespace janus {

XTensorDistService::XTensorDistService(int32_t global_rank,
                                       int32_t world_size,
                                       const torch::Device& device)
    : global_rank_(global_rank),
      world_size_(world_size),
      device_(device),
      initialized_(false) {}

void XTensorDistService::Hello(::google::protobuf::RpcController* controller,
                               const proto::Status* request,
                               proto::Status* response,
                               ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);
  if (!initialized_) {
    ctrl->SetFailed("Server is not initialized");
  } else {
    response->set_ok(true);
  }
}

void XTensorDistService::GetMemoryInfo(
    ::google::protobuf::RpcController* controller,
    const proto::Status* request,
    proto::MemoryInfoResponse* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);

    Device device(device_);
    device.set_device();

    // Empty torch cache to get accurate memory info
    int32_t device_id = device_.index();

    const auto available_memory = device.free_memory();
    const auto total_memory = device.total_memory();

    // Update device monitor
    DeviceMonitor::get_instance().set_total_memory(device_id, total_memory);

    LOG(INFO) << "GetMemoryInfo: global_rank=" << global_rank_
              << ", available_memory=" << available_memory
              << ", total_memory=" << total_memory;

    // Returns 0 for both fields on failure (handled by caller)
    response->set_available_memory(available_memory);
    response->set_total_memory(total_memory);
  });
}

void XTensorDistService::InitPhyPagePool(
    ::google::protobuf::RpcController* controller,
    const proto::InitPhyPagePoolRequest* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);

    int64_t num_pages = request->num_pages();
    LOG(INFO) << "InitPhyPagePool: global_rank=" << global_rank_
              << ", num_pages=" << num_pages;

    try {
      // Initialize PhyPagePool with specified number of pages
      PhyPagePool::get_instance().init(device_, num_pages);

      // Initialize GlobalXTensor after PhyPagePool
      GlobalXTensor::get_instance().init(device_);
      LOG(INFO) << "GlobalXTensor initialized on worker " << global_rank_;

      response->set_ok(true);
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to init PhyPagePool/GlobalXTensor: " << e.what();
      response->set_ok(false);
    }
  });
}

void XTensorDistService::MapToKvTensors(
    ::google::protobuf::RpcController* controller,
    const proto::KvTensorRequest* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);

    std::string model_id = request->model_id();

    // Convert proto offsets to vector
    std::vector<offset_t> offsets;
    offsets.reserve(request->offsets_size());
    for (int i = 0; i < request->offsets_size(); ++i) {
      offsets.push_back(request->offsets(i));
    }

    // Call XTensorAllocator to map
    auto& allocator = XTensorAllocator::get_instance();
    bool success = allocator.map_to_kv_tensors(model_id, offsets);
    response->set_ok(success);
  });
}

void XTensorDistService::UnmapFromKvTensors(
    ::google::protobuf::RpcController* controller,
    const proto::KvTensorRequest* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);

    std::string model_id = request->model_id();

    // Convert proto offsets to vector
    std::vector<offset_t> offsets;
    offsets.reserve(request->offsets_size());
    for (int i = 0; i < request->offsets_size(); ++i) {
      offsets.push_back(request->offsets(i));
    }

    // Call XTensorAllocator to unmap
    auto& allocator = XTensorAllocator::get_instance();
    bool success = allocator.unmap_from_kv_tensors(model_id, offsets);
    response->set_ok(success);
  });
}

void XTensorDistService::AllocWeightPages(
    ::google::protobuf::RpcController* controller,
    const proto::AllocWeightPagesRequest* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);

    std::string model_id = request->model_id();
    size_t num_pages = request->num_pages();

    LOG(INFO) << "AllocWeightPages: model_id=" << model_id
              << ", num_pages=" << num_pages;

    auto& pool = PhyPagePool::get_instance();
    auto& allocator = XTensorAllocator::get_instance();

    // Try contiguous allocation first (from GlobalXTensor)
    page_id_t start_page = pool.allocate_contiguous_from_right(num_pages);
    if (start_page >= 0) {
      allocator.record_weight_allocation(model_id, start_page, num_pages);
      response->set_ok(true);
      LOG(INFO) << "AllocWeightPages success: model_id=" << model_id
                << ", start_page=" << start_page << ", num_pages=" << num_pages;
      return;
    }

    // Fallback: try non-contiguous allocation using XTensor
    LOG(WARNING) << "Contiguous allocation failed for " << num_pages
                 << " pages, trying non-contiguous fallback (XTensor)";

    std::vector<page_id_t> page_ids = pool.allocate_pages_from_right(num_pages);
    if (page_ids.empty()) {
      LOG(ERROR) << "Failed to allocate " << num_pages
                 << " weight pages (both contiguous and non-contiguous)";
      response->set_ok(false);
      return;
    }

    allocator.record_weight_fallback_allocation(model_id, page_ids);
    response->set_ok(true);
    LOG(INFO) << "AllocWeightPages success (fallback): model_id=" << model_id
              << ", num_pages=" << num_pages;
  });
}

void XTensorDistService::FreeWeightPages(
    ::google::protobuf::RpcController* controller,
    const proto::FreeWeightPagesRequest* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);

    std::string model_id = request->model_id();

    LOG(INFO) << "FreeWeightPages: model_id=" << model_id;

    // Free weight pages via XTensorAllocator (frees pages in PhyPagePool)
    auto& allocator = XTensorAllocator::get_instance();
    size_t num_freed = allocator.free_weight(model_id);

    response->set_ok(num_freed > 0);

    LOG(INFO) << "FreeWeightPages: freed " << num_freed << " pages for model "
              << model_id;
  });
}

void XTensorDistService::GetXTensorOffsets(
    ::google::protobuf::RpcController* controller,
    const proto::GetXTensorOffsetsRequest* request,
    proto::GetXTensorOffsetsResponse* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);

    std::string model_id = request->model_id();
    uint64_t block_size_bytes = request->block_size_bytes();

    // Convert proto block_ids to vector
    std::vector<int32_t> block_ids;
    block_ids.reserve(request->block_ids_size());
    for (int i = 0; i < request->block_ids_size(); ++i) {
      block_ids.push_back(request->block_ids(i));
    }

    auto& allocator = XTensorAllocator::get_instance();
    if (!allocator.is_initialized()) {
      LOG(ERROR) << "XTensorAllocator not initialized on worker";
      return;
    }

    std::vector<XTensorLayerOffsets> layer_offsets;
    XTensorBlockBytes block_bytes;
    if (!allocator.get_xtensor_offsets(
            /*dp_rank=*/0,
            model_id,
            block_ids,
            block_size_bytes,
            layer_offsets,
            &block_bytes)) {
      LOG(ERROR) << "Failed to get xtensor offsets for model " << model_id;
      return;
    }

    for (const auto& layer_offsets_item : layer_offsets) {
      auto* layer_offsets_proto = response->add_layer_offsets();
      for (const auto k_offset : layer_offsets_item.k_offsets) {
        layer_offsets_proto->add_k_offsets(k_offset);
      }
      for (const auto v_offset : layer_offsets_item.v_offsets) {
        layer_offsets_proto->add_v_offsets(v_offset);
      }
      for (const auto index_offset : layer_offsets_item.index_offsets) {
        layer_offsets_proto->add_index_offsets(index_offset);
      }
    }
    auto* block_bytes_proto = response->mutable_block_bytes();
    block_bytes_proto->set_k_block_bytes(block_bytes.k_block_bytes);
    block_bytes_proto->set_v_block_bytes(block_bytes.v_block_bytes);
    block_bytes_proto->set_index_block_bytes(block_bytes.index_block_bytes);

    VLOG(1) << "GetXTensorOffsets: model_id=" << model_id
            << ", num_blocks=" << block_ids.size()
            << ", num_layers=" << layer_offsets.size();
  });
}

}  // namespace janus
