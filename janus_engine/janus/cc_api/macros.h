#pragma once

namespace janus {

#ifdef JANUS_CAPI_WEAK
#define JANUS_CAPI_EXPORT \
  __attribute__((visibility("default"))) __attribute((weak))
#else
#define JANUS_CAPI_EXPORT __attribute__((visibility("default")))
#endif  // JANUS_CAPI_WEAK

}  // namespace janus