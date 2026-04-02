#include "janus_server_registry.h"

namespace janus {

JanusServer* ServerRegistry::register_server(const std::string& name) {
  {
    LOG(INFO) << "Register server " << name << ".";

    std::lock_guard<std::mutex> lock(mutex_);
    if (servers_.find(name) != servers_.end()) {
      LOG(ERROR) << "Register server failed, " << name
                 << " has been registered already.";
      return servers_[name].get();
    }

    servers_[name] = JanusServerFactory::create_janus_server();
    return servers_[name].get();
  }
}

void ServerRegistry::unregister_server(const std::string& name) {
  {
    LOG(INFO) << "Unregister server " << name << ".";

    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = servers_.find(name);
    if (iter == servers_.end()) {
      LOG(ERROR) << "Unregister server failed, " << name
                 << " isn't registered.";
    } else {
      servers_.erase(iter);
    }
  }
}

JanusServer* ServerRegistry::get_server(const std::string& name) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (servers_.find(name) == servers_.end()) {
      LOG(ERROR) << "Server " << name << " doesn't exist.";
      return nullptr;
    }

    return servers_[name].get();
  }
}

JanusServer* ServerRegistry::try_get_server(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = servers_.find(name);
  if (it == servers_.end()) {
    return nullptr;
  }
  return it->second.get();
}

}  // namespace janus
