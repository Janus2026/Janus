# NOTE: Set VCPKG_ROOT before running this script.
# Example:
#   git clone https://github.com/microsoft/vcpkg.git "$HOME/vcpkg"
#   export VCPKG_ROOT="$HOME/vcpkg"
#   if gtest is not installed, install it
#  /vcpkg-src/vcpkg install gtest
mkdir -p build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_UNITY=ON -DENABLE_PCH=ON
cmake --build build -j $(nproc)