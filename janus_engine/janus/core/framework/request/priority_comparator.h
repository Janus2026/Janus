#pragma once
#include <functional>
#include <memory>
#include <string>

#include "common.pb.h"
#include "framework/request/request.h"

namespace janus {
class PriorityComparator {
 public:
  virtual bool operator()(const std::shared_ptr<Request>& a,
                          const std::shared_ptr<Request>& b) const = 0;
  virtual ~PriorityComparator() = default;
};

struct FCFSComparator : public PriorityComparator {
  bool operator()(const std::shared_ptr<Request>& a,
                  const std::shared_ptr<Request>& b) const override;
};

struct StrictPriorityComparator : public PriorityComparator {
  bool operator()(const std::shared_ptr<Request>& a,
                  const std::shared_ptr<Request>& b) const override;
};

struct DensityComparator : public PriorityComparator {
  bool operator()(const std::shared_ptr<Request>& a,
                  const std::shared_ptr<Request>& b) const override;
};

struct DeadlineComparator : public PriorityComparator {
  bool operator()(const std::shared_ptr<Request>& a,
                  const std::shared_ptr<Request>& b) const override;
};

struct SJFComparator : public PriorityComparator {
  bool operator()(const std::shared_ptr<Request>& a,
                  const std::shared_ptr<Request>& b) const override;
};

struct DecodeDeadlineComparator : public PriorityComparator {
  bool operator()(const std::shared_ptr<Request>& a,
                  const std::shared_ptr<Request>& b) const override;
};

struct DensityWithAntiStarveComparator : public PriorityComparator {
  bool operator()(const std::shared_ptr<Request>& a,
                  const std::shared_ptr<Request>& b) const override;
};

struct DecodeDensityWithAntiStarveComparator : public PriorityComparator {
  bool operator()(const std::shared_ptr<Request>& a,
                  const std::shared_ptr<Request>& b) const override;
};

struct DecodeDensityComparator : public PriorityComparator {
  bool operator()(const std::shared_ptr<Request>& a,
                  const std::shared_ptr<Request>& b) const override;
};

struct UrgencyDensityComparator : public PriorityComparator {
  bool operator()(const std::shared_ptr<Request>& a,
                  const std::shared_ptr<Request>& b) const override;
};

struct UrgencyPriorityComparator : public PriorityComparator {
  bool operator()(const std::shared_ptr<Request>& a,
                  const std::shared_ptr<Request>& b) const override;
};

struct DecodeUrgencyDensityComparator : public PriorityComparator {
  bool operator()(const std::shared_ptr<Request>& a,
                  const std::shared_ptr<Request>& b) const override;
};

std::function<bool(const std::shared_ptr<Request>&,
                   const std::shared_ptr<Request>&)>
create_comparator(const std::string& priority_strategy, bool reverse = false);

}  // namespace janus
