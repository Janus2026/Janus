#include "mm_input.h"

#include <glog/logging.h>

#include "mm_handler.h"

namespace janus {

MMInputTransfer::MMInputTransfer() {
  mm_handlers_ = std::make_unique<MMHandlerSet>();
}

MMInputTransfer::~MMInputTransfer() {}

MMErrCode MMInputTransfer::trans(const std::vector<Message>& messages,
                                 MMInput& inputs) {
  inputs.clear();
  std::vector<MMInputItem> ins;

  for (int idx = 0; idx < messages.size(); ++idx) {
    const auto& message = messages[idx];
    const auto& mmc = std::get<MMContentVec>(message.content);

    MMErrCode code = this->trans(mmc, ins, inputs.payload());
    if (code != MMErrCode::SUCCESS) {
      return code;
    }

    inputs.insert(ins);
  }
  return MMErrCode::SUCCESS;
}

MMErrCode MMInputTransfer::trans(const MMContentVec& mmc,
                                 std::vector<MMInputItem>& inputs,
                                 MMPayload& payload) {
  inputs.clear();
  for (int idx = 0; idx < mmc.size(); ++idx) {
    const auto& item = mmc[idx];
    const auto& type = item.type;

    if (type != "text") {
      MMInputItem input;
      MMErrCode code = mm_handlers_->process(type, item, input, payload);
      if (code != MMErrCode::SUCCESS) {
        return code;
      }

      inputs.emplace_back(std::move(input));
    }
  }

  return MMErrCode::SUCCESS;
}

}  // namespace janus
