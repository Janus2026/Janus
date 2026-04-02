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

#include <filesystem>
#include <string>

namespace janus::kernel::npu {

inline std::string GetBinaryDir() {
#ifdef TEST_BINARY_DIR
  std::string binary_dir = TEST_BINARY_DIR;
  if (!binary_dir.empty() && binary_dir.back() != '/') {
    binary_dir += "/";
  }
  return binary_dir;
#else
  return "";
#endif
}

inline std::string GetKernelBinaryPath(const std::string& filename) {
  std::string binary_dir = GetBinaryDir();
  return binary_dir + filename;
}

inline bool FileExists(const std::string& filepath) {
  return std::filesystem::exists(filepath) &&
         std::filesystem::is_regular_file(filepath);
}

inline bool KernelBinaryExists(const std::string& filename) {
  return FileExists(GetKernelBinaryPath(filename));
}

}  // namespace janus::kernel::npu
