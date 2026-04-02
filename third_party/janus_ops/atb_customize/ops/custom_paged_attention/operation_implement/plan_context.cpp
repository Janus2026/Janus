#include "custom_paged_attention_function.h"
#include "atb/context.h"
#include "atb/context/context_base.h"
#include <memory>

namespace atb {
namespace customize {

constexpr uint64_t TILING_BUFFER_BLOCK_SIZE = 1024 * 1024 * 3;

class PlanContext : public ContextBase {
 public:
  virtual uint8_t *GetHostTilingBuffer() override {
    if (!hostTilingBuffer_) {
      void *addr = nullptr;
      Status st = aclrtMallocHost(&addr, TILING_BUFFER_BLOCK_SIZE);
      if (st != 0) {
          ATB_LOG(ERROR) << "PlanContext::GetHostTilingBuffer aclrtMallocHost failed!, error code: " << st;
          return nullptr;
      }
      hostTilingBuffer_.reset(reinterpret_cast<uint8_t *>(addr));
    }
    return hostTilingBuffer_.get();
  }
 private:
  static void AclHostMemoryDeleter(uint8_t* ptr) {
    if (ptr != nullptr) {
      Status st = aclrtFreeHost(ptr);
      if (st != 0) {
        ATB_LOG(ERROR) << "PlanContext::AclHostDeleter aclrtFreeHost failed!, error code: " << st;
      }
    }
  }
  std::unique_ptr<uint8_t, decltype(&AclHostMemoryDeleter)> hostTilingBuffer_{nullptr, AclHostMemoryDeleter};
};

Status CreatePlanContext(Context **context)
{
    if (!context) {
        ATB_LOG(ERROR) << "param context is null, CreateContext fail";
        return ERROR_INVALID_PARAM;
    }

    ContextBase *contextBase = new (std::nothrow) PlanContext();
    if (!contextBase) {
        ATB_LOG(ERROR) << "new ContextBase fail, CreateContext fail";
        return ERROR_OUT_OF_HOST_MEMORY;
    }

    Status st = contextBase->Init();
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << "ContextBase init fail, CreateContext fail";
        delete contextBase;
        contextBase = nullptr;
        return st;
    }

    *context = contextBase;
    return NO_ERROR;
}
} // namespace customize
} // namespace atb
