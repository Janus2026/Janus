#pragma once

#include <string>

namespace janus {

class InstanceName {
 public:
  static InstanceName* name() {
    static InstanceName n;
    return &n;
  }

  void set_name(const std::string& name) {
    name_ = name;
    name_hash_ = std::to_string(std::hash<std::string>{}(name_));
  }

  std::string get_name() const { return name_; }

  std::string get_name_hash() const { return name_hash_; }

 private:
  InstanceName() {}
  InstanceName(const InstanceName&) = delete;
  InstanceName& operator=(const InstanceName&) = delete;

 private:
  std::string name_;
  std::string name_hash_;
};

}  // namespace janus