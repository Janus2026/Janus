#pragma once

#include <folly/Function.h>

#include <functional>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "framework/request/dit_request_params.h"
#include "master.h"
#include "scheduler/dit_scheduler.h"

namespace janus {

class Call;

class DiTMaster : public Master {
 public:
  explicit DiTMaster(const Options& options);
  ~DiTMaster();

  // handle a request, the engine will execute the request asynchronously
  void handle_request(DiTRequestParams params,
                      std::optional<Call*> call,
                      DiTOutputCallback callback);

  // batch generation
  void handle_batch_request(std::vector<DiTRequestParams> params_vec,
                            BatchDiTOutputCallback callback);

  // start running loop
  void run() override;

  // generate will run all request done at once,
  // this is a blocking call
  void generate();

 private:
  std::unique_ptr<DiTEngine> engine_;

  std::unique_ptr<DiTScheduler> scheduler_;

  // thread pool for handling requests
  std::unique_ptr<ThreadPool> threadpool_;

  // thread for moving forward the scheduler
  std::thread loop_thread_;

  // flag to stop the loop
  std::atomic_bool stoped_{false};

  // flag to indicate if the handler is running
  std::atomic_bool running_{false};
};

}  // namespace janus
