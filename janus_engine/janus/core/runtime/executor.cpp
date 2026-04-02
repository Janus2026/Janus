#include "executor.h"

#include "common/global_flags.h"
#include "executor_impl_factory.h"
#include "platform/device.h"

namespace janus {

Executor::Executor(CausalLM* model,
                   const ModelArgs& args,
                   const torch::Device& device,
                   const runtime::Options& options) {
  std::string backend = (options.backend() != "vlm" && FLAGS_enable_graph)
                            ? Device::type_str()
                            : options.backend();
  impl_ = ExecutorImplFactory::get_instance().create_executor_impl(
      model, args, device, options, backend);
}

ForwardInput Executor::prepare_inputs(Batch& batch) {
  return impl_->prepare_inputs(batch);
}

ModelOutput Executor::forward(const torch::Tensor& tokens,
                              const torch::Tensor& positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelInputParams& params) {
  return impl_->run(tokens, positions, kv_caches, params);
}

}  // namespace janus
