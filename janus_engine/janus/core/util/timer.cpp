#include "timer.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>

namespace janus {

Timer::Timer() : start_(absl::Now()) {}

// reset the timer
void Timer::reset() { start_ = absl::Now(); }

// get the elapsed time in seconds
double Timer::elapsed_seconds() const {
  return absl::ToDoubleSeconds(absl::Now() - start_);
}

// get the elapsed time in milliseconds
double Timer::elapsed_milliseconds() const {
  return absl::ToDoubleMilliseconds(absl::Now() - start_);
}

// get the elapsed time in microseconds
double Timer::elapsed_microseconds() const {
  return absl::ToDoubleMicroseconds(absl::Now() - start_);
}
}  // namespace janus