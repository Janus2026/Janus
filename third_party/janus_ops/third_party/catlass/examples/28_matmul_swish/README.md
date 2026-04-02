# MatmulSwish Example Readme
## 代码组织
```
├── 28_matmul_swish
│   ├── CMakeLists.txt   # CMake编译文件
│   ├── README.md
│   └── matmul_swish.cpp # 主文件
```
## 功能介绍

Swish:
$$
Swish(x) = x \cdot Sigmoid(x)
$$
Sigmoid：
$$
Sigmoid(x)=\frac{1}{1+e^{-x}}
$$
因此计算函数为：
$$
x = a \times b\\
Sigmoid(out)=\frac{x}{1+e^{-x}}
$$


## 使用示例

- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 执行算子
```
# 编译指定用例
bash scripts/build.sh 28_matmul_swish
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./28_matmul_swish 256 512 1024 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```