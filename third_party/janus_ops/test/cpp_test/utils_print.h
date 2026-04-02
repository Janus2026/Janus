#pragma once

#include <iostream>
#include "utils_tensor.h"
void print_info(aclTensor* weight) {
    printf("------------------------- Format --------------------------\n");
    std::cout << weight->GetStorageFormat() << " " << weight->GetOriginalFormat() << " " << weight->GetViewFormat() << std::endl;
    
    printf("------------------------- Original Shape --------------------------\n");
    std::cout << weight->GetOriginalShape().GetDimNum() << " " << weight->GetOriginalShape().GetShapeSize() << std::endl;
    for (int i = 0; i < weight->GetOriginalShape().GetDimNum(); i++) {
        std::cout << weight->GetOriginalShape().GetDim(i) << " ";
    }
    std::cout << std::endl;

    printf("------------------------- view Strides --------------------------\n");
    std::cout << weight->GetViewShape().GetDimNum() << " " << weight->GetViewShape().GetShapeSize() << std::endl;
    for (int i = 0; i < weight->GetViewShape().GetDimNum(); i++) {
        std::cout << weight->GetViewShape().GetDim(i) << " ";
    }
    std::cout << std::endl;
    
    printf("------------------------- Storage Format --------------------------\n");
    std::cout << weight->GetStorageShape().GetDimNum() << " " << weight->GetStorageShape().GetShapeSize() << std::endl;
    for (int i = 0; i < weight->GetStorageShape().GetDimNum(); i++) {
        std::cout << weight->GetStorageShape().GetDim(i) << " ";
    }
    std::cout << std::endl;
    
    
    printf("------------------------- View Strides --------------------------\n");
    for (int i = 0; i < weight->GetViewStrides().size(); i++) {
        std::cout << weight->GetViewStrides().data()[i] << " ";
    }
    std::cout << std::endl;
}

void print_tensor_layout(const torch::Tensor& tensor) {
    std::cout << "=== Tensor Layout ===" << std::endl;
    std::cout << "Shape:      " << tensor.sizes() << std::endl;
    std::cout << "Strides:    " << tensor.strides() << std::endl;
    std::cout << "Contiguous: " << tensor.is_contiguous() << std::endl;
    
    if (tensor.is_contiguous(at::MemoryFormat::Contiguous)) {
        std::cout << "C-contiguous (row major)" << std::endl;
    } else if (tensor.is_contiguous(at::MemoryFormat::ChannelsLast)) {
        std::cout << "Channels-last (NHWC)" << std::endl;
    } else {
        std::cout << "Non-standard layout" << std::endl;
    }
    
    std::cout << "Dtype:      " << tensor.dtype() << std::endl;
    std::cout << "Device:     " << tensor.device() << std::endl;
    std::cout << "Data addr:  " << tensor.data_ptr() << std::endl;
}