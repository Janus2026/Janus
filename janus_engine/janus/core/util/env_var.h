#pragma once

#include <optional>
#include <string>

namespace janus {
namespace util {

bool get_bool_env(const std::string& key, bool defaultValue);

// Get an integer value from an environment variable.
// Returns the default value if the environment variable is not set or cannot be
// parsed.
int64_t get_int_env(const std::string& key, int64_t defaultValue);

std::string get_string_env(const std::string& name);

// Get the timeout in seconds for process group test operations.
// This timeout is used when waiting for process group initialization tests
// to complete in multi-device/multi-node scenarios. The default value is 4
// seconds, but can be overridden by setting the
// JANUS_PROCESS_GROUP_ASYNC_TIMEOUT_SECONDS environment variable. This is
// particularly useful in multi-node multi-device communication scenarios
// where network latency may cause the default 4-second timeout to be
// insufficient.
int64_t get_process_group_test_timeout_seconds();

// Returns an optional fixed acceptance rate for speculative decoding (for
// performance debugging only). If the JANUS_FIX_SPECULATIVE_ACCEPTANCE_RATE
// environment variable is set to a value in [0.0, 1.0], returns
// std::optional<double> with that value; otherwise returns std::nullopt.
// WARNING: Using this will influence model accuracy and should not be used in
// production.
std::optional<double> get_fix_speculative_acceptance_rate();

}  // namespace util
}  // namespace janus
