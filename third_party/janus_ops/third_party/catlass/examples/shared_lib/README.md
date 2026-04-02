# 打包为共享库

有时我们希望在已有的成熟工程中添加模板库算子，实现加速计算的效果，但又不希望大幅度改变构建工程。为此我们可以将模板库算子编译成共享库，以方便在已有工程中调用。

## 代码结构

```bash
examples/shared_lib
├── include
│   └── catlass_kernel.h            # 头文件
└── src
    ├── common
    │   └── common.hpp          # 公共头文件，预留为多个kernel中的模板函数共用
    ├── host                    # host侧接口
    │   ├── basic_matmul.cpp    
    │   └── ...
    └── kernel                  # kernel侧算子
        ├── basic_matmul.hpp
        └── ...
```

## 编译产物结构

```bash
output/shared_lib
├── include
│   └── catlass_kernel.h # 头文件
└── lib
    ├── libcatlass_kernel.a # 静态链接库
    └── libcatlass_kernel.so # 动态链接库
```

## 使用说明

假设待添加算子为`custom_matmul`。

### 算子kernel实现

分别增加如下文件和代码：

- 在`src/kernel`文件夹中创建`custom_matmul.hpp`，实现算子本身。

```cpp
#include "catlass/catlass.hpp"
// catlass头文件...

using namespace Catlass;

template <
    class LayoutA,
    class LayoutB,
    class LayoutC
>
CATLASS_GLOBAL
void custom_matmul(
    GemmCoord problemShape,
    GM_ADDR gmA, LayoutA layoutA,
    GM_ADDR gmB, LayoutB layoutB,
    GM_ADDR gmC, LayoutC layoutC
    // 按需定义输入参数...
)
{
    // 使用Catlass api定义算子...
}
```

- 在`src/host`文件夹中创建`custom_matmul.cpp`，实现host接口，该接口负责整理输入后使用`内核调用符`调用算子。

```cpp
// ...
void CustomMatmul(uint32_t blockNum, aclrtStream stream, ernelInfo kernelInfo) {
    Catlass::GemmCoord problemShape{kernelInfo.m, kernelInfo.n, kernelInfo.k};
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;
    LayoutA layoutA{kernelInfo.m, kernelInfo.k};
    LayoutB layoutB{kernelInfo.k, kernelInfo.n};
    LayoutC layoutC{kernelInfo.m, kernelInfo.n};
    custom_matmul<<<blockNum, nullptr, stream>>>(problemShape,
        kernelInfo.inputAddr.at(0), layoutA,
        kernelInfo.inputAddr.at(1), layoutB,
        kernelInfo.outputAddr.at(0), layoutC);
}
// ...
```

参数意义如下：

| 参数名       | 类型          | 作用                                                |
| ------------ | ------------- | --------------------------------------------------- |
| `blockNum`   | `uint32_t`    | 设定aiCore个数                                      |
| `stream`     | `aclrtStream` | NPU流                                               |
| `kernelInfo` | `KernelInfo`  | 算子执行的数据地址和输入详细情况，如mnk等维度的大小 |

可根据实际需要自行修改参数。

- 在`include/catlass_kernel.h`中增加`custom_matmul.cpp`中的host入口，以供外部调用。

```cpp
// ...
void CustomMatmul(uint32_t blockNum, aclrtStream stream, ernelInfo kernelInfo);
// ...
```

- 在`CMakeLists.txt`增加`catlass_add_kernel(custom_matmul dav-c220 ${CMAKE_CURRENT_SOURCE_DIR}/src/host/custom_matmul.cpp)`编译命令。

- 如果你增加了多个算子，但又存在相同定义的`模板函数`，这种情况在链接阶段会提示重复符号。为解决这个问题，你可以将这类函数以`inline`形式存入公共的`common`路径中。

### 编译

```bash
bash scripts/build.sh shared_lib
```

## 注意事项

- 我们目前提供了三种典型算子作为示例：
  - `BasicMatmul`：基本矩阵乘法，并实现了类型模板的实现方法
  - `GroupedMatmul`：分组矩阵乘法，提供分组输入输出示例
  - `OptimizedMatmul`：优化矩阵乘法，提供CV融合的示例
- 本节是算子打包成动态库的一个示例，可根据需要自行扩展功能，并不仅局限于已有的代码。
