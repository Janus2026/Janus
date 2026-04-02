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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <type_traits>
#include <vector>

namespace janus::kernel::npu {

constexpr size_t CONST_8 = 8;
constexpr size_t CONST_4 = 4;

class ArgsBuilder final {
 public:
  ArgsBuilder() { buf_.reserve(256); }

  const void* data() const { return buf_.data(); }
  size_t size() const { return size_; }

  void clear() {
    buf_.clear();
    size_ = 0;
  }

  template <typename... Ts>
  void constructArgs(const Ts&... args) {
    (pack_one(args), ...);
  }

  template <typename T>
  void add(const T& v) {
    static_assert(std::is_trivially_copyable_v<T>,
                  "T must be trivially copyable");
    add_aligned<T>(v, alignof(T));
  }

  template <typename T>
  void add_aligned(const T& v, size_t alignment) {
    static_assert(std::is_trivially_copyable_v<T>,
                  "T must be trivially copyable");
    pad_to(alignment);
    ensure_capacity(sizeof(T));
    std::memcpy(buf_.data() + size_, &v, sizeof(T));
    size_ += sizeof(T);
  }

  void pad_to(size_t alignment) {
    if (alignment == 0) {
      return;
    }
    const size_t mod = size_ % alignment;
    if (mod == 0) {
      return;
    }
    const size_t pad = alignment - mod;
    ensure_capacity(pad);
    std::memset(buf_.data() + size_, 0, pad);
    size_ += pad;
  }

  std::string hex_dump() const {
    std::ostringstream oss;
    oss << std::hex << std::uppercase << std::setfill('0');
    for (size_t i = 0; i < size_; ++i) {
      if (i > 0 && i % 16 == 0) {
        oss << '\n';
      }
      oss << std::setw(2) << (static_cast<unsigned>(buf_[i]) & 0xFF);
      if (i % 16 != 15 && i < size_ - 1) {
        oss << ' ';
      }
    }
    return oss.str();
  }

  std::string hex_dump_compact() const {
    std::ostringstream oss;
    oss << std::hex << std::uppercase << std::setfill('0');
    for (size_t i = 0; i < size_; ++i) {
      oss << std::setw(2) << (static_cast<unsigned>(buf_[i]) & 0xFF);
    }
    return oss.str();
  }

  bool compare_with(const void* other_data, size_t other_size) const {
    if (size_ != other_size) {
      return false;
    }
    return std::memcmp(buf_.data(), other_data, size_) == 0;
  }

  template <typename StructType>
  bool compare_with_struct(const StructType& s) const {
    static_assert(std::is_trivially_copyable_v<StructType>,
                  "StructType must be trivially copyable");
    return compare_with(&s, sizeof(StructType));
  }

  struct DiffInfo {
    size_t first_diff_offset;
    uint8_t expected_byte;
    uint8_t actual_byte;
    bool is_match;
    bool is_nullptr_field;
    std::string message;
  };

  bool is_nullptr_at_offset(size_t offset) const {
    if (offset + sizeof(void*) > size_) {
      return false;
    }
    void* ptr_value = nullptr;
    std::memcpy(&ptr_value, buf_.data() + offset, sizeof(void*));
    return ptr_value == nullptr;
  }

  bool is_nullptr_at_offset(const void* data,
                            size_t data_size,
                            size_t offset) const {
    if (offset + sizeof(void*) > data_size) {
      return false;
    }
    void* ptr_value = nullptr;
    std::memcpy(
        &ptr_value, static_cast<const uint8_t*>(data) + offset, sizeof(void*));
    return ptr_value == nullptr;
  }

  DiffInfo diff_with(const void* other_data, size_t other_size) const {
    DiffInfo info;
    info.is_match = true;
    info.is_nullptr_field = false;
    const size_t min_size = std::min(size_, other_size);
    const uint8_t* other_bytes = static_cast<const uint8_t*>(other_data);
    const uint8_t* this_bytes = reinterpret_cast<const uint8_t*>(buf_.data());

    for (size_t i = 0; i < min_size; ++i) {
      if (this_bytes[i] != other_bytes[i]) {
        info.is_match = false;
        info.first_diff_offset = i;
        info.expected_byte = other_bytes[i];
        info.actual_byte = this_bytes[i];

        const size_t aligned_offset = (i / sizeof(void*)) * sizeof(void*);
        const bool this_is_null = is_nullptr_at_offset(aligned_offset);
        const bool other_is_null =
            is_nullptr_at_offset(other_data, other_size, aligned_offset);

        if (this_is_null || other_is_null) {
          info.is_nullptr_field = true;
          std::ostringstream oss;
          oss << "Difference at offset " << i << " (aligned to "
              << aligned_offset << "): ";
          if (this_is_null && other_is_null) {
            oss << "Both are nullptr, but bytes differ (likely "
                   "padding/uninitialized)";
          } else if (this_is_null) {
            oss << "This is nullptr, other is non-null";
          } else {
            oss << "This is non-null, other is nullptr";
          }
          info.message = oss.str();
        } else {
          std::ostringstream oss;
          oss << "Difference at offset " << i;
          info.message = oss.str();
        }
        return info;
      }
    }

    if (size_ != other_size) {
      info.is_match = false;
      info.first_diff_offset = min_size;
      info.expected_byte = (size_ < other_size) ? other_bytes[min_size] : 0;
      info.actual_byte = (size_ > other_size) ? this_bytes[min_size] : 0;
      std::ostringstream oss;
      oss << "Size mismatch: this=" << size_ << ", other=" << other_size;
      info.message = oss.str();
    }

    return info;
  }

  template <typename StructType>
  DiffInfo diff_with_struct(const StructType& s) const {
    static_assert(std::is_trivially_copyable_v<StructType>,
                  "StructType must be trivially copyable");
    return diff_with(&s, sizeof(StructType));
  }

  std::string debug_info() const {
    std::ostringstream oss;
    oss << "ArgsBuilder Debug Info:\n";
    oss << "  Total size: " << size_ << " bytes\n";
    oss << "  Buffer capacity: " << buf_.capacity() << " bytes\n";

    oss << "  Pointer fields (8-byte aligned, checking for nullptr):\n";
    for (size_t i = 0; i + sizeof(void*) <= size_; i += sizeof(void*)) {
      if (i % 8 == 0) {
        void* ptr_value = nullptr;
        std::memcpy(&ptr_value, buf_.data() + i, sizeof(void*));
        oss << "    [" << std::setw(4) << std::setfill('0') << std::hex << i
            << std::dec << "] ";
        if (ptr_value == nullptr) {
          oss << "void* = nullptr (0x0000000000000000)";
        } else {
          oss << "void* = 0x" << std::hex << std::setw(16) << std::setfill('0')
              << reinterpret_cast<uintptr_t>(ptr_value) << std::dec;
        }
        oss << "\n";
      }
    }

    oss << "  Hex dump (first 64 bytes):\n";
    const size_t dump_size = std::min(size_, size_t(64));
    for (size_t i = 0; i < dump_size; i += 16) {
      oss << "    [" << std::setw(4) << std::setfill('0') << std::hex << i
          << std::dec << "] ";
      for (size_t j = 0; j < 16 && (i + j) < dump_size; ++j) {
        oss << std::setw(2) << std::setfill('0') << std::hex
            << (static_cast<unsigned>(buf_[i + j]) & 0xFF) << " ";
      }
      oss << "\n";
    }
    if (size_ > 64) {
      oss << "    ... (truncated, total " << size_ << " bytes)\n";
    }
    return oss.str();
  }

 private:
  void ensure_capacity(size_t additional) {
    const size_t required = size_ + additional;
    if (required > buf_.capacity()) {
      buf_.reserve(std::max(required, buf_.capacity() * 2));
    }
    if (required > buf_.size()) {
      buf_.resize(required);
    }
  }

 private:
  template <typename T>
  static constexpr size_t packed_alignment() {
    using U = std::remove_cv_t<T>;
    if constexpr (std::is_pointer_v<U>) {
      return CONST_8;
    } else if constexpr (std::is_same_v<U, double> ||
                         std::is_same_v<U, int64_t> ||
                         std::is_same_v<U, uint64_t>) {
      return CONST_8;
    } else if constexpr (std::is_same_v<U, float> ||
                         std::is_same_v<U, int32_t> ||
                         std::is_same_v<U, uint32_t>) {
      return CONST_4;
    } else if constexpr (std::is_integral_v<U> || std::is_floating_point_v<U>) {
      return alignof(U);
    } else {
      return alignof(U);
    }
  }

  template <typename T>
  void pack_one(const T& v) {
    using U = std::remove_cv_t<T>;
    if constexpr (std::is_pointer_v<U>) {
      void* p = const_cast<void*>(reinterpret_cast<const void*>(v));
      add_aligned<void*>(p, packed_alignment<U>());
    } else if constexpr (std::is_integral_v<U> || std::is_floating_point_v<U>) {
      add_aligned<U>(v, packed_alignment<U>());
    } else {
      static_assert(std::is_trivially_copyable_v<U>,
                    "T must be trivially copyable");
      add_aligned<U>(v, packed_alignment<U>());
    }
  }

  std::vector<char> buf_;
  size_t size_ = 0;
};

}  // namespace janus::kernel::npu
