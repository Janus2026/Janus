#ifndef INCLUDE_COMMON_H
#define INCLUDE_COMMON_H

#define CONST_2 2

#define SET_FLAG(trigger, waiter, e) AscendC::SetFlag<AscendC::HardEvent::trigger##_##waiter>((e))
#define WAIT_FLAG(trigger, waiter, e) AscendC::WaitFlag<AscendC::HardEvent::trigger##_##waiter>((e))
#define PIPE_BARRIER(pipe) AscendC::PipeBarrier<PIPE_##pipe>()

#ifndef __force_inline__
#define __force_inline__ inline __attribute__((always_inline))
#endif

#endif
