#pragma once

#include <mutex>
#include <unordered_map>

#include "janus_server.h"

namespace janus {

class ServerRegistry {
 public:
  static ServerRegistry& get_instance() {
    static ServerRegistry instance;
    return instance;
  }

  JanusServer* register_server(const std::string& name);
  void unregister_server(const std::string& name);
  JanusServer* get_server(const std::string& name);
  JanusServer* try_get_server(const std::string& name);

 private:
  ServerRegistry() = default;
  ~ServerRegistry() = default;
  DISALLOW_COPY_AND_ASSIGN(ServerRegistry);

  std::unordered_map<std::string, std::unique_ptr<JanusServer>> servers_;
  std::mutex mutex_;
};

}  // namespace janus
