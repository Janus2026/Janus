#include <cmath>

namespace AtbOps {
namespace Utils {
template<class T>
class Compare {
public:
    static bool IsEqual(const T &lh, const T &rh)
    {
        return lh == rh;
    }
};

template<>
class Compare<float> {
public:
    static bool IsEqual(const float &lh, const float &rh)
    {
        return std::abs(lh - rh) < 0.000001; // float precise 0.000001
    }
};

template<>
class Compare<double> {
public:
    static bool IsEqual(const double &lh, const double &rh)
    {
        return std::abs(lh - rh) < 0.000000001; // double precise 0.000000001
    }
};
} // namespace AtbOps
} //Utils
