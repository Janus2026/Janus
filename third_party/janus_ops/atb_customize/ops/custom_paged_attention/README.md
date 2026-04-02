# Custom Paged Attention Operator

This is a custom version of the Paged Attention operator, created for implementing custom Paged Attention functionality.

## Directory Structure

```
custom_paged_attention/
├── kernel_implement/           # Kernel implementation
│   ├── custom_paged_attention_operation.cpp
│   ├── custom_paged_attention_kernel.cpp
│   ├── CMakeLists.txt
│   ├── include/                # Header files
│   ├── tiling/                 # Tiling related files
│   │   ├── custom_paged_attention_tiling.cpp
│   │   ├── custom_paged_attention_tiling.h
│   │   ├── custom_paged_attention_tiling_dependency.cpp
│   │   └── custom_paged_attention_tiling_dependency.h
│   └── op_kernel/              # Operator kernel files
│       ├── custom_paged_attention_decoder_nd_common.cce
│       ├── custom_paged_attention_mask_mix.cce
│       └── utils/              # Utility files
│           ├── common.h
│           └── kernel/         # Kernel utilities
│               ├── common_func.h
│               ├── common.h
│               ├── hardware.h
│               ├── iterator.h
│               ├── kernel_utils.h
│               ├── layout.h
│               ├── mem.h
│               ├── mma.h
│               ├── set_fpc.h
│               ├── simd.h
│               ├── utils.h
│               └── iterators/  # Iterator implementations
│                   ├── gm_to_l1_iterator.inc
│                   ├── gm_to_ub_iterator.inc
│                   ├── l0c_to_gm_iterator.inc
│                   ├── l0c_to_l1_iterator.inc
│                   ├── l0c_to_ub_iterator.inc
│                   ├── l1_to_bt_iterator.inc
│                   ├── l1_to_fb_iterator.inc
│                   ├── l1_to_l0_iterator.inc
│                   └── l1_to_ub_iterator.inc
├── operation_implement/        # Operation implementation
│   ├── custom_paged_attention_operation.h
│   ├── custom_paged_attention_operation.cpp
│   ├── custom_paged_attention_ops_runner.h
│   ├── custom_paged_attention_ops_runner.cpp
│   ├── custom_paged_attention_ops_runner_910a.h
│   ├── custom_paged_attention_ops_runner_910a.cpp
│   ├── custom_paged_attention_runner_utils.h
│   ├── custom_paged_attention_runner_utils.cpp
│   ├── custom_paged_attention_param.h
│   └── custom_paged_attention_param.cpp
└── tests/                      # Test files
    ├── CMakeLists.txt
    ├── custom_paged_attention_test.cpp
    ├── device_version_check.h
    ├── device_version_check.cpp
    ├── test_util.h
    └── data/                   # Test data generation
        ├── paged_attention_gen.py
        └── unpad_flashattention_encoder_gen.py
```

## Key Features

1. **Class Renaming**: All class names renamed from `PagedAttention*` to `CustomPagedAttention*`
2. **File Renaming**: All file names prefixed with `custom_`
3. **Header Guard Macros**: Updated to `CUSTOM_PAGED_ATTENTION_*` format
4. **Include Paths**: Updated all include paths to use new file names
5. **Comprehensive Testing**: Includes comparison tests between custom and standard implementations
6. **Utility Functions**: Provides fp16 conversion and comparison utilities

## Testing

The operator includes comprehensive testing capabilities:

- **Comparison Testing**: Tests custom implementation against standard `atb::infer::PagedAttentionParam`
- **Data Validation**: Uses fp16 data comparison with tolerance checking
- **Device-Host Data Transfer**: Includes utilities for copying data between device and host
- **Random Data Generation**: Generates test data with proper context length distribution

### Test Features

- **Custom vs Standard Comparison**: Validates that custom implementation produces same results as standard
- **FP16 Data Handling**: Proper conversion and comparison of half-precision floating point data
- **Memory Management**: Proper allocation and cleanup of device and host memory
- **Error Handling**: Comprehensive error checking and reporting

## Usage

This custom operator can be used for:
- Implementing custom Paged Attention algorithms
- Adding specific optimizations or modifications
- Learning and experimentation purposes
- Performance comparison with standard implementations

## Building

Use the ops_customize build system to compile this custom operator:

```bash
cd ops_customize
./build.sh
```

## Running Tests

To run the comparison tests:

```bash
cd ops_customize
./build.sh unittest
```

The test will:
1. Create both custom and standard PagedAttention operations
2. Execute both with identical input data
3. Compare the results using fp16 data comparison
4. Report whether the implementations produce aligned results
