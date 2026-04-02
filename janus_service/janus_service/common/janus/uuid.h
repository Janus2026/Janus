#pragma once
#include <absl/random/random.h>

#include <string>

namespace janus_service {
namespace llm {

class ShortUUID {
 public:
  ShortUUID() = default;

  std::string random(size_t len = 0);

 private:
  std::string alphabet_ =
      "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
      "abcdefghijkmnopqrstuvwxyz";
  absl::BitGen gen_;
};

}  // namespace llm
}  // namespace janus_service
