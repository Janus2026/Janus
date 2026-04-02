#pragma once

#include <torch/torch.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "mm_item_state.h"
#include "mm_type.h"

namespace janus {

using MMKey = std::string;
using MMValue = std::variant<torch::Tensor, std::vector<torch::Tensor>>;
using MMDict = std::unordered_map<MMKey, MMValue>;

class MMDataItem {
  using MMMetadata = std::variant<ImageMetadata, VideoMetadata, AudioMetadata>;

 public:
  class IVisitor {
   public:
    virtual ~IVisitor() = default;
    virtual bool visit(MMDataItem& item) = 0;
  };

 public:
  MMDataItem(MMType type);
  MMDataItem(MMType type, const MMDict& data);
  MMDataItem(MMType type, const MMDict& data, const MMMetadata& metadata);

  bool valid() const { return type_ != MMType::NONE; }
  bool is_type(MMType type) const { return type_ == type; }

  const MMDict& data() const { return data_; }
  MMDict& mutable_data() { return data_; }
  void set_data(const MMDict& data) { data_ = std::move(data); }

  MMType type() const { return type_; }
  bool has(const MMKey& key) const;

  void get(const MMKey& key, std::vector<torch::Tensor>& vec) const;
  bool is_embedded() const { return has("embedding"); }

  template <typename T>
  std::optional<T> get(const MMKey& key) const {
    if (!valid()) return std::nullopt;

    const auto& itor = data_.find(key);
    return itor != data_.end() ? std::optional<T>(std::get<T>(itor->second))
                               : std::nullopt;
  }

  template <typename T>
  bool add(const MMKey& key, const T& value) {
    auto [iter, success] = data_.insert({key, value});
    return success;
  }

  template <typename T>
  std::optional<T> get_metadata() const {
    if (!valid()) return std::nullopt;

    if (std::holds_alternative<T>(metadata_)) {
      return std::get<T>(metadata_);
    } else {
      return std::nullopt;
    }
  }

  template <typename T>
  void set_metadata(const T& meta) {
    metadata_ = meta;
  }

  const MMItemState& state() const { return state_; }
  MMItemState& mutable_state() { return state_; }

  const MMMetadata& metadata() const { return metadata_; }
  void debug_print() const;

 private:
  MMType type_ = MMType::NONE;
  MMDict data_;
  MMMetadata metadata_;
  MMItemState state_;
};

}  // namespace janus