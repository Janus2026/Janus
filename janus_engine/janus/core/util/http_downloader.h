#pragma once

#include <brpc/channel.h>
#include <brpc/controller.h>
#include <bthread/rwlock.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace janus {

class HttpDownloader {
 public:
  HttpDownloader() = default;
  virtual ~HttpDownloader() {}

  bool fetch_data(const std::string& url, std::string& data);

 protected:
  bool parse_url(const std::string& url, std::string& host);
  virtual bool download(const std::string& host,
                        const std::string& url,
                        std::string& data) = 0;
};

class BRpcDownloader : public HttpDownloader {
 public:
  BRpcDownloader() = default;
  ~BRpcDownloader() = default;

  bool download(const std::string& host,
                const std::string& url,
                std::string& data) override;

 private:
  std::shared_ptr<brpc::Channel> get_channel(const std::string& host);

 private:
  inline static bthread::RWLock rw_lock_;
  inline static std::unordered_map<std::string, std::shared_ptr<brpc::Channel>>
      channels_;
};

}  // namespace janus
