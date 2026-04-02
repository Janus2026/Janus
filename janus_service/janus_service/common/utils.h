#pragma once

#include <string>

namespace janus_service {
namespace utils {

bool enable_debug_log();
bool is_port_available(int port);
bool get_bool_env(const std::string& key, bool defaultValue);
std::string get_local_ip();

}  // namespace utils
}  // namespace janus_service
