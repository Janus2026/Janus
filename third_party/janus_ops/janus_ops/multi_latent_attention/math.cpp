#include "math.h"
#include <sys/time.h>
#include <cstdlib>
#include <cstddef>
#include <cstdio>

namespace AtbOps {
namespace Utils {
size_t RoundUp(size_t size, size_t divisor)
{
    if (divisor == 0 || (size + divisor - 1) < size) {
        printf("divisor is 0 or (size + divisor - 1) < size");
        return size;
    }
    return (size + divisor - 1) / divisor * divisor;
}

size_t RoundDown(size_t size, size_t divisor)
{
    if (divisor == 0) {
        return size;
    }
    return size / divisor * divisor;
}

void SetRandseeds(uint32_t &randseed)
{
    if (randseed == 0xffffffff) {
        struct timeval tv;
        gettimeofday(&tv, nullptr);
        randseed = static_cast<uint32_t>(tv.tv_sec * 1000 + tv.tv_usec / 1000); // 1000 convert in us/ms/s
    }
    srand(randseed);
}
} // namespace Utils {
} // namespace AtbOps {
