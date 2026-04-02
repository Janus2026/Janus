#pragma once

#include <xxHash/xxhash.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

namespace janus {

constexpr uint32_t XXH3_128BITS_HASH_VALUE_LEN = sizeof(XXH128_hash_t);

struct XXH3Key {
  uint8_t data[XXH3_128BITS_HASH_VALUE_LEN];

  XXH3Key() {}
  XXH3Key(const uint8_t* const input_data) {
    std::memcpy(data, input_data, XXH3_128BITS_HASH_VALUE_LEN);
  }

  bool operator==(const XXH3Key& other) {
    return std::memcmp(reinterpret_cast<const char*>(data),
                       reinterpret_cast<const char*>(other.data),
                       XXH3_128BITS_HASH_VALUE_LEN);
  }

  void set(const uint8_t* const input_data) {
    std::memcpy(data, input_data, XXH3_128BITS_HASH_VALUE_LEN);
  }

  std::string debug_string() const {
    std::string rt;
    for (int i = 0; i < XXH3_128BITS_HASH_VALUE_LEN; i++) {
      rt += std::to_string(int64_t(data[i])) + " ";
    }

    return rt;
  }
};

struct FixedStringKeyHash {
  size_t operator()(const XXH3Key& key) const {
    return std::hash<std::string_view>()(std::string_view(
        reinterpret_cast<const char*>(key.data), sizeof(key.data)));
  }
};

struct FixedStringKeyEqual {
  bool operator()(const XXH3Key& left, const XXH3Key& right) const {
    return std::strncmp(reinterpret_cast<const char*>(left.data),
                        reinterpret_cast<const char*>(right.data),
                        sizeof(left.data)) == 0;
  }
};

}  // namespace janus
