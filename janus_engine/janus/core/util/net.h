#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace janus {
namespace net {

std::string get_local_ip_addr();
int get_local_free_port();
uint64_t convert_ip_port_to_uint64(const std::string& ip, uint16_t port);
std::pair<std::string, uint16_t> convert_uint64_to_ip_port(uint64_t input);
void parse_host_port_from_addr(const std::string& addr,
                               std::string& host,
                               int& port);

std::string extract_ip(const std::string& input);
std::string extract_port(const std::string& input);
}  // namespace net
}  // namespace janus
