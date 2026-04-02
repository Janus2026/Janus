#pragma once

#include <absl/container/flat_hash_set.h>
#if defined(USE_NPU)
#include <hccl/hccl.h>
#endif

#include <unordered_map>

#include "common/macros.h"
#include "worker.pb.h"

namespace janus {

class CollectiveService : public proto::Collective {
 public:
  CollectiveService(int dp_group_num, int total_num, int device_idx);
  virtual ~CollectiveService() = default;

  void Sync(::google::protobuf::RpcController* controller,
            const proto::AddressInfo* request,
            proto::CommUniqueIdList* response,
            ::google::protobuf::Closure* done) override;

  // Wait for all workers to connect.
  // timeout_sec > 0: throw std::runtime_error if not all connected in time.
  // timeout_sec == 0: wait indefinitely (legacy behavior).
  std::unordered_map<int32_t, std::string> wait(int timeout_sec = 0);

 private:
  DISALLOW_COPY_AND_ASSIGN(CollectiveService);

#if defined(USE_NPU)
  void to_proto_list(const std::vector<HcclRootInfo>& src,
                     proto::CommUniqueIdList* dst);
  void from_proto_list(const proto::CommUniqueIdList& src,
                       std::vector<HcclRootInfo>* dst);
#endif

 private:
  int total_num_ = 0;
#if defined(USE_NPU)
  std::vector<HcclRootInfo> root_infos_;
#endif
  std::mutex mutex_;
  std::unordered_map<int32_t, std::string> addrs_map_;
};

}  // namespace janus
