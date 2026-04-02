/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 */

#ifndef ATB_SPEED_LAYERS_PARALLEL_LMHEAD_ALLTOALL_H
#define ATB_SPEED_LAYERS_PARALLEL_LMHEAD_ALLTOALL_H
#include <atb/atb_infer.h>

#include "operations/fusion/lmhead/lmhead.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed {
namespace common {
atb::Status ParallelLmHeadAllToAll(const LmHeadParam &param, atb::Operation **operation);
} // namespace common
} // namespace atb_speed
#endif