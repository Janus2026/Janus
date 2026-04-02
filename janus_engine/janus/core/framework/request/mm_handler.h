#pragma once

#include <string>
#include <vector>

#include "core/common/message.h"
#include "core/common/types.h"
#include "mm_input.h"

namespace janus {

struct MMInputItem;
struct MMPayload;
struct MMInput;

class MMHandlerBase {
 public:
  MMHandlerBase() = default;
  virtual ~MMHandlerBase() = default;

  MMErrCode process(const MMContent& content,
                    MMInputItem& input,
                    MMPayload& payload);

  virtual MMErrCode load(const MMContent& content,
                         MMInputItem& input,
                         MMPayload& payload) = 0;

  virtual MMErrCode decode(MMInputItem& input) = 0;

 protected:
  MMErrCode load_from_dataurl(const std::string& url,
                              std::string& data,
                              MMPayload& payload);

  MMErrCode load_from_local(const std::string& url, std::string& data);

  MMErrCode load_from_http(const std::string& url, std::string& data);

 protected:
  std::string httpurl_prefix_{"http"};
};

class ImageHandler : public MMHandlerBase {
 public:
  ImageHandler() = default;
  ~ImageHandler() = default;

  virtual MMErrCode load(const MMContent& content,
                         MMInputItem& input,
                         MMPayload& payload) override;
  virtual MMErrCode decode(MMInputItem& input) override;

 private:
  std::string dataurl_prefix_{"data:image"};
};

class VideoHandler : public MMHandlerBase {
 public:
  VideoHandler() = default;
  ~VideoHandler() = default;

  virtual MMErrCode load(const MMContent& content,
                         MMInputItem& input,
                         MMPayload& payload) override;
  virtual MMErrCode decode(MMInputItem& input) override;

 private:
  std::string dataurl_prefix_{"data:video"};
};

class AudioHandler : public MMHandlerBase {
 public:
  AudioHandler() = default;
  ~AudioHandler() = default;

  virtual MMErrCode load(const MMContent& content,
                         MMInputItem& input,
                         MMPayload& payload) override;
  virtual MMErrCode decode(MMInputItem& input) override;

 private:
  std::string dataurl_prefix_{"data:audio"};
};

class MMHandlerSet {
 public:
  MMHandlerSet();
  ~MMHandlerSet();

  MMErrCode process(const std::string& type,
                    const MMContent& content,
                    MMInputItem& input,
                    MMPayload& payload);

 private:
  std::unordered_map<std::string, std::unique_ptr<MMHandlerBase>> handlers_;
};

}  // namespace janus
