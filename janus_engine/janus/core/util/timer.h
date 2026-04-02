#pragma once

#include <absl/time/time.h>

namespace janus {

class Timer final {
 public:
  Timer();

  // reset the timer
  void reset();

  // get the elapsed time.
  double elapsed_seconds() const;
  double elapsed_milliseconds() const;
  double elapsed_microseconds() const;

 private:
  // the start time of the timer
  absl::Time start_;
};

}  // namespace janus