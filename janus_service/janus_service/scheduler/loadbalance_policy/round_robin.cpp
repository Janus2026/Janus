#include "round_robin.h"

namespace janus_service {

bool RoundRobin::select_instances_pair(std::shared_ptr<Request> request) {
  return instance_mgr_->get_next_instance_pair(request->model, &request->routing);
}

}  // namespace janus_service
