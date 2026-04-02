include(cc_library)

# ===========================================================================
# xxHash library build configuration
# ===========================================================================
#
# xxHash supports multiple SIMD backends (auto-selected per platform):
#
#   x86 / x86_64  ── runtime dispatch via xxh_x86dispatch.c
#                     (SSE2 / AVX2 / AVX512, detected with CPUID at startup)
#
#   ARM / AArch64 ── compile-time detection built into xxhash.h
#                     (NEON / SVE, enabled by -march flags or defaults)
#
#   Other         ── portable scalar fallback
#
# The dispatch module (xxh_x86dispatch.c) MUST be compiled with *default*
# target flags — it uses __attribute__((target("avx2"))) etc. internally and
# will #error on non-x86 platforms.
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. Detect target architecture
#    Use CMAKE_SYSTEM_PROCESSOR (respects cross-compilation toolchain files).
# ---------------------------------------------------------------------------
set(XXHASH_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "xxHash: target architecture = ${XXHASH_ARCH}")

set(XXHASH_SRCS xxhash.c)
set(XXHASH_HDRS xxhash.h xxh3.h)

# ---------------------------------------------------------------------------
# 2. Platform-specific source selection
# ---------------------------------------------------------------------------
if(XXHASH_ARCH MATCHES "(x86_64|amd64|AMD64|i[3-6]86)")
  # x86 / x86_64: add runtime SIMD dispatcher.
  list(APPEND XXHASH_SRCS xxh_x86dispatch.c)
  list(APPEND XXHASH_HDRS xxh_x86dispatch.h)
  message(STATUS "xxHash: x86 dispatch enabled (SSE2 / AVX2 / AVX512)")

elseif(XXHASH_ARCH MATCHES "(aarch64|arm64|ARM64|armv[7-9])")
  # ARM / AArch64: NEON & SVE are activated automatically by xxhash.h
  # when the compiler defines __ARM_NEON / __ARM_FEATURE_SVE.
  # No extra source files needed.
  #
  # However, we must ensure the compiler flags actually enable these features.
  # - NEON is mandatory on AArch64 (always defined), but -march=armv8-a is
  #   the minimal baseline.
  # - SVE requires -march=armv8.2-a+sve or newer.
  # - Using -march=native on the build host picks the best available ISA.
  # ---------------------------------------------------------------------------
  # Try -march=armv8.2-a+dotprod+crypto (good for Kunpeng 920, Neoverse N1, etc.)
  # Fall back to -march=native (best when building on target hardware).
  # ---------------------------------------------------------------------------
  include(CheckCCompilerFlag)
  check_c_compiler_flag("-march=armv8.2-a+dotprod+crypto" XXHASH_HAS_MARCH_CUSTOMIZATION)
  if(XXHASH_HAS_MARCH_CUSTOMIZATION)
    set(XXHASH_ARM_COPTS "-march=armv8.2-a+dotprod+crypto")
    message(STATUS "xxHash: ARM using -march=armv8.2-a+dotprod+crypto")
  else()
    set(XXHASH_ARM_COPTS "-march=native")
    message(STATUS "xxHash: ARM using -march=native (fallback)") 
  endif()

  message(STATUS "xxHash: ARM platform (NEON / SVE via compile-time detection)")

else()
  message(STATUS "xxHash: generic platform — scalar fallback")
endif()

# ---------------------------------------------------------------------------
# 3. Build the library
# ---------------------------------------------------------------------------
cc_library(
  NAME
    xxHash
  HDRS
    ${XXHASH_HDRS}
  SRCS
    ${XXHASH_SRCS}
  INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
)
