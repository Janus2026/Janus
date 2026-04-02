## Overview

This sample introduces the high-performance implementation of the replace token operator and provides the operator project calling method.

## Directory structure introduction

```
├──replace_token     // Call the ReplaceToken custom operator using kernel function direct tuning
│   └── replace_token       // Operator Engineering
│   └── test            // This directory is the unit test of replace token, including accuracy test and performance test
```

## Operator Description

The operator implements functions
- During the scheduling process, when each step's computation is completed, the input for the next step needs to be updated. This process can be performed asynchronously on the NPU to improve efficiency. 



The calculation formula of Replace-Token is:

```c++
  auto& flatten_tokens = inputs.token_ids;
  auto neg_mask = (flatten_tokens < 0);
  auto clamped_neg_indices = torch::clamp(-flatten_tokens, 0);
  auto replacement = last_step_output_.sample_output.next_tokens.index(
      {clamped_neg_indices - 1});
  inputs.token_ids = torch::where(neg_mask, replacement, flatten_tokens);
```

## Start 

```
cd replace_token
bash build.sh
```

## Test

```
cd replace_test
bash run.sh
```
