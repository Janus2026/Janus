#pragma once

#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace janus {

class SharedMemoryManager {
 public:
  explicit SharedMemoryManager(const std::string& name,
                               size_t size,
                               bool& is_creator);
  virtual ~SharedMemoryManager();
  void* base_address() const { return addr_; }
  int64_t size() const { return size_; }
  std::string name() const { return shm_name_; }

 private:
  std::string shm_name_;
  int fd_ = -1;
  void* addr_ = MAP_FAILED;
  int64_t size_ = 0;
  int64_t current_offset_ = 0;

  static void cleanup_handler(int sig);
  static std::vector<std::string> pending_cleanups;
  static std::mutex cleanup_mutex;
};

}  // namespace janus