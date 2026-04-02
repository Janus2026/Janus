#pragma once

#include <torch/torch.h>

#include "common/macros.h"
#include "executor_impl.h"
#include "framework/model/causal_lm.h"
#include "options.h"

namespace janus {

class ExecutorImplFactory {
 public:
  using Creator =
      std::function<std::unique_ptr<ExecutorImpl>(CausalLM*,
                                                  const ModelArgs&,
                                                  const torch::Device&,
                                                  const runtime::Options&)>;

  static ExecutorImplFactory& get_instance();

  bool register_creator(const std::string& name, Creator creator);

  std::unique_ptr<ExecutorImpl> create_executor_impl(
      CausalLM* model,
      const ModelArgs& args,
      const torch::Device& device,
      const runtime::Options& options,
      const std::string& backend);

  DISALLOW_COPY_AND_ASSIGN(ExecutorImplFactory);

 private:
  ExecutorImplFactory() = default;

  ~ExecutorImplFactory() = default;

  std::unordered_map<std::string, Creator> creators_;
};

#define REGISTER_EXECUTOR(backend, class_type)                                 \
  namespace {                                                                  \
  bool class_type##_registered = []() -> bool {                                \
    return ExecutorImplFactory::get_instance().register_creator(               \
        backend,                                                               \
        [](CausalLM* model,                                                    \
           const ModelArgs& args,                                              \
           const torch::Device& device,                                        \
           const runtime::Options& options) -> std::unique_ptr<ExecutorImpl> { \
          return std::make_unique<class_type>(model, args, device, options);   \
        });                                                                    \
  }();                                                                         \
  }

}  // namespace janus
