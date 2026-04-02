/* Copyright 2026 The Janus Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 */

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif
// rt kernel
typedef struct {
  uint32_t magic{0};
  uint32_t version{0};
  const void* data{nullptr};
  uint64_t length{0};
} RtDevBinaryT;

typedef void* rtStream_t;
typedef int32_t rtError_t;

int rtFunctionRegister(void* binHandle,
                       const void* subFunc,
                       const char* stubName,
                       const void* kernelInfoExt,
                       uint32_t funcMode);
int rtDevBinaryRegister(const RtDevBinaryT* bin, void** hdl);
int rtDevBinaryUnRegister(void* hdl);
int rtKernelLaunch(const void* stubFunc,
                   uint32_t blockDim,
                   void* args,
                   uint32_t argsSize,
                   void* smDesc,
                   rtStream_t sm);
// rt other
int rtGetC2cCtrlAddr(uint64_t* addr, uint32_t* len);

constexpr uint32_t RT_DEV_BINARY_MAGIC_ELF_AICUBE = 0x41494343U;
constexpr uint32_t RT_DEV_BINARY_MAGIC_ELF_AIVEC = 0x41415246U;
constexpr uint32_t RT_DEV_BINARY_MAGIC_ELF = 0x43554245U;
constexpr int32_t RT_ERROR_NONE = 0;

#ifdef __cplusplus
}
#endif

namespace janus::kernel::npu {

using KernelStubHandle = void*;

class KernelRegistry {
 public:
  static KernelRegistry& get_instance();

  KernelRegistry(const KernelRegistry&) = delete;
  KernelRegistry& operator=(const KernelRegistry&) = delete;

  bool register_kernel(const std::string& kernel_name,
                       const std::string& binary_path);

  // return kernel stub handle, which is used to call the kernel function
  KernelStubHandle get_kernel_stub(const std::string& kernel_name) const;

  bool is_kernel_registered(const std::string& kernel_name) const;

  void cleanup();

  bool parse_json_config(const std::string& json_path,
                         std::string& kernel_name,
                         std::string& mix_mode,
                         int64_t& workspace_size,
                         int64_t& lock_init_value,
                         int64_t& lock_num);

  bool get_kernel_workspace_config(const std::string& kernel_name,
                                   int64_t& workspace_size,
                                   int64_t& lock_init_value,
                                   int64_t& lock_num) const;

 private:
  KernelRegistry() = default;
  ~KernelRegistry();

  struct KernelInfo {
    std::string name;
    char* buffer = nullptr;                // binary buffer
    KernelStubHandle stub_func = nullptr;  // rtFunctionRegister funcstub handle
    void* bin_handle = nullptr;            // binary bin handle
    std::string persistent_func_name;
    std::string mix_mode;
    int64_t workspace_size;
    int64_t lock_init_value;
    int64_t lock_num;
  };

  char* load_binary_file(const std::string& file_path, uint32_t& file_size);
  bool register_binary(KernelInfo& info, uint32_t binary_size);

  std::unordered_map<std::string, KernelInfo> kernel_infos_;
};

}  // namespace janus::kernel::npu
