#include "executor_impl_factory.h"

#include "platform/device.h"
#include "runtime/base_executor_impl.h"
#include "runtime/vlm_executor_impl.h"
#if defined(USE_NPU)
#include "runtime/acl_graph_executor_impl.h"
#elif defined(USE_MLU)
#include "runtime/mlu_graph_executor_impl.h"
#elif defined(USE_CUDA)
#include "runtime/cuda_graph_executor_impl.h"
#endif

namespace janus {

ExecutorImplFactory& ExecutorImplFactory::get_instance() {
  static ExecutorImplFactory instance;
  return instance;
}

bool ExecutorImplFactory::register_creator(const std::string& name,
                                           Creator creator) {
  auto [it, inserted] = creators_.emplace(name, std::move(creator));
  return inserted;
}

std::unique_ptr<ExecutorImpl> ExecutorImplFactory::create_executor_impl(
    CausalLM* model,
    const ModelArgs& args,
    const torch::Device& device,
    const runtime::Options& options,
    const std::string& backend) {
  auto it = creators_.find(backend);
  if (it == creators_.end()) {
    throw std::runtime_error("No valid backend found: " + backend);
  }

  return it->second(model, args, device, options);
}

}  // namespace janus
