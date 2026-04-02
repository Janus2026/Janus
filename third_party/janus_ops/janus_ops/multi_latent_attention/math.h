#include <cstdint>
#include <cstddef>
namespace AtbOps {
namespace Utils {
template<typename T>
T CeilDiv(T dividend, T divisor)
{
    return (divisor == 0) ? 0 : ((dividend + divisor - 1) / divisor);
}

size_t RoundUp(size_t size, size_t divisor = 32);

size_t RoundDown(size_t size, size_t divisor = 32);

void SetRandseeds(uint32_t &randseed);
} // namespace Utils
} // namespace AtbOps
