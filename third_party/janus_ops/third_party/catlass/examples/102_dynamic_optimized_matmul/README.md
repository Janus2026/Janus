# OptimizedMatmul Example Readme
## 代码组织
```
examples/102_dynamic_optimized_matmul
├── CMakeLists.txt
├── README.md
├── dynamic_optimized_matmul.cpp
├── impl
│   ├── kernel
│   │   └── common_matmul_kernel.h
│   ├── scripts
│   │   ├── templates
│   │   │   ├── common_matmul_template.py
│   │   │   └── launch_map_template.py
│   │   ├── utils
│   │   │   └── config.py
│   │   └── wrapper_code_gen.py
│   └── wrapper # 自动生成
└── include
    ├── launch_map.h # 自动生成
    ├── do_tiling_b16.h
    ├── dynamic_optimized_matmul.h
    ├── platform_info.h
    ├── select_kernel_b16.h
    ├── tiling_params.h
    └── utils.h
```
## 工程说明
工程默认编译成静态库，如果想编译成动态库，请把CMakeLists.txt中的`STATIC`改为`SHARED`，并手动export动态库路径。
工程编译前会调用python脚本生成代码，具体包括调用各模板的外围代码，以及launch_map.h(包含tilingKey和具体Kernel的映射关系)
如果需要进行批量性能测试，请注释掉精度比较代码，由于精度比较使用cpu算golden，耗时较长。
DynamicOptimizedMamtul根据shape动态确定Tiling参数，并尽力选择最好的模板进行计算，尽力获取最优性能，但是不保证是最优性能。
## 使用示例
- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/quickstart.md#算子编译)
- 执行算子
```
# 编译指定用例
bash scripts/build.sh 102_dynamic_optimized_matmul
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|LayoutA|LayoutB|Device ID
# 0 is RowMajor, 1 is ColumnMajor
./102_dynamic_optimized_matmul 256 512 1024 0 1 0
```
执行结果如下，说明精度比对成功。
```
Compare success.
```