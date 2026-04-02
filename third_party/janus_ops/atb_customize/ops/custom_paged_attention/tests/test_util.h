#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <cstdlib>
#include "acl/acl.h"
#include "atb/operation.h"
#include "atb/types.h"
#include "atb/atb_infer.h"
#define CHECK_STATUS(status) \
 do { \
 if ((status) != 0) { \
 std::cout << __FILE__ << ":" << __LINE__ << " [error]: " << (status) << std::endl; \
 exit(1); \
 } \
 } while (0)
#define CHECK_STATUS_EXPR(status, expr) \
 do { \
 if ((status) != 0) { \
 std::cout << __FILE__ << ":" << __LINE__ << " [error]: " << (status) << std::endl; \
 expr; \
 } \
 } while (0)
/**
 * @brief Create a Tensor object
 * @param dataType Data type
 * @param format Data format
 * @param shape Data shape
 * @return atb::Tensor Returns the created Tensor object
 */
atb::Tensor CreateTensor(const aclDataType dataType, const aclFormat format, std::vector<int64_t>
shape)
{
 atb::Tensor tensor;
 tensor.desc.dtype = dataType;
 tensor.desc.format = format;
 tensor.desc.shape.dimNum = shape.size();
 // Set tensor dims to elements in shape
 for (size_t i = 0; i < shape.size(); i++) {
  tensor.desc.shape.dims[i] = shape.at(i);
 }
 tensor.dataSize = atb::Utils::GetTensorSize(tensor); // Calculate Tensor data size
 CHECK_STATUS(aclrtMalloc(&tensor.deviceData, tensor.dataSize,
aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST));
 return tensor;
}
/**
 * @brief Create a Tensor object from existing buffer
 * @param dataType Data type
 * @param format Data format
 * @param shape Data shape
 * @param buffer Existing buffer pointer
 * @param dataSize Size of the buffer (optional, if 0 will calculate from tensor)
 * @return atb::Tensor Returns the created Tensor object using existing buffer
 */
atb::Tensor CreateTensorFromBuffer(const aclDataType dataType, const aclFormat format, 
                                   std::vector<int64_t> shape, void *buffer, 
                                   uint64_t dataSize = 0)
{
 atb::Tensor tensor;
 tensor.desc.dtype = dataType;
 tensor.desc.format = format;
 tensor.desc.shape.dimNum = shape.size();
 // Set tensor dims to elements in shape
 for (size_t i = 0; i < shape.size(); i++) {
  tensor.desc.shape.dims[i] = shape.at(i);
 }
 tensor.dataSize = (dataSize > 0) ? dataSize : atb::Utils::GetTensorSize(tensor);
 tensor.deviceData = buffer; // Use existing buffer instead of allocating
 return tensor;
}
/**
 * @brief Perform data type conversion, call Elewise cast Op
 * @param contextPtr Context pointer
 * @param stream Stream
 * @param inTensor Input tensor
 * @param outTensorType Output tensor data type
 * @param shape Output tensor shape
 * @return atb::Tensor Converted tensor
 */
atb::Tensor CastOp(atb::Context *contextPtr, aclrtStream stream, const atb::Tensor inTensor,
 const aclDataType outTensorType, std::vector<int64_t> shape)
{
 uint64_t workspaceSize = 0;
 void *workspace = nullptr;
 // Create Elewise ELEWISE_CAST
 atb::infer::ElewiseParam castParam;
 castParam.elewiseType = atb::infer::ElewiseParam::ELEWISE_CAST;
 castParam.outTensorType = outTensorType;
 atb::Operation *castOp = nullptr;
 CHECK_STATUS(CreateOperation(castParam, &castOp));
 atb::Tensor outTensor = CreateTensor(outTensorType, aclFormat::ACL_FORMAT_ND, shape); // cast output tensor
 atb::VariantPack castVariantPack; // Parameter pack
 castVariantPack.inTensors = {inTensor};
 castVariantPack.outTensors = {outTensor};
 // Validate input and output tensors when calling Setup interface
 CHECK_STATUS(castOp->Setup(castVariantPack, workspaceSize, contextPtr));
 if (workspaceSize > 0) {
 CHECK_STATUS(aclrtMalloc(&workspace, workspaceSize,
aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST));
 }
 // Execute ELEWISE_CAST
 CHECK_STATUS(castOp->Execute(castVariantPack, (uint8_t *)workspace, workspaceSize,
contextPtr));
 CHECK_STATUS(aclrtSynchronizeStream(stream)); // Stream synchronization, wait for device side task completion
 if (workspaceSize > 0) {
 CHECK_STATUS(aclrtFree(workspace)); // Clean up workspace
 }
 return outTensor;
}
/**
 * @brief Simple wrapper, copy vector data to create tensor
 * @details Used to create ACL_FLOAT16/ACL_FLOAT32 type tensors
 * @param contextPtr Context pointer
 * @param stream Stream
 * @param data Input vector data
 * @param outTensorType Expected output tensor data type
 * @param format Output tensor format, i.e. NZ, ND, etc.
 * @param shape Output tensor shape
 * @return atb::Tensor Returns the created tensor
 */
template <typename T>
atb::Tensor CreateTensorFromVector(atb::Context *contextPtr, aclrtStream stream, std::vector<T> data,
 const aclDataType outTensorType, const aclFormat format, std::vector<int64_t> shape)
{
 atb::Tensor tensor;
 aclDataType intermediateType;
 switch (outTensorType) {
 case aclDataType::ACL_FLOAT:
 case aclDataType::ACL_FLOAT16:
 case aclDataType::ACL_BF16:
 intermediateType = aclDataType::ACL_FLOAT;
 break;
 case aclDataType::ACL_INT32:
 case aclDataType::ACL_INT64:
 intermediateType = aclDataType::ACL_INT32;
 break;
 default:
 intermediateType = outTensorType;
 }
 tensor = CreateTensor(intermediateType, format, shape);
 CHECK_STATUS(aclrtMemcpy(
 tensor.deviceData, tensor.dataSize, data.data(), sizeof(T) * data.size(),
ACL_MEMCPY_HOST_TO_DEVICE));
 if (intermediateType == outTensorType) {
 // Original created tensor type, no conversion needed
 return tensor;
 }
 return CastOp(contextPtr, stream, tensor, outTensorType, shape);
}

/**
 * @brief Copy tensor device data to host
 * @param tensor Tensor to be copied, will modify its hostData field
 * @return Status Operation status
 */
atb::Status CopyDeviceToHost(atb::Tensor &tensor)
{
    if (tensor.deviceData == nullptr) {
        return -1; // ERROR_INVALID_TENSOR_ADDR
    }
    
    // Allocate host memory
    if (tensor.hostData == nullptr) {
        tensor.hostData = malloc(tensor.dataSize);
        if (tensor.hostData == nullptr) {
            return -1; // ERROR_OUT_OF_HOST_MEMORY
        }
    }
    
    // Copy from device to host
    aclError ret = aclrtMemcpy(tensor.hostData, tensor.dataSize, 
                               tensor.deviceData, tensor.dataSize, 
                               ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        return -1; // ERROR_COPY_HOST_MEMORY_FAIL
    }
    
    return 0; // NO_ERROR
}

// Tolerance constants for float16 data comparison
constexpr float ATOL = 0.001f;
constexpr float RTOL = 0.001f;

/**
 * @brief Compare if two float16 data are equal
 * @param a First float16 value
 * @param b Second float16 value
 * @param atol Absolute tolerance
 * @param rtol Relative tolerance
 * @return bool Whether they are equal
 */
// Simple fp16 to fp32 conversion method
float HalfToFloat(uint16_t h)
{
    // Use union to avoid strict aliasing rule issues
    union {
        uint32_t i;
        float f;
    } converter;
    
    // Method 1: Use bit operations for conversion
    uint32_t sign = (h & 0x8000) << 16;  // Sign bit
    uint32_t exp = (h & 0x7C00) >> 10;   // Exponent bit
    uint32_t mantissa = h & 0x03FF;      // Mantissa bit
    
    if (exp == 0) {
        // Zero or denormalized number
        if (mantissa == 0) {
            // Zero
            converter.i = sign;
            return converter.f;
        } else {
            // Denormalized number
            exp = 127 - 15; // Adjust bias
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                exp--;
            }
            mantissa &= 0x3FF; // Remove implicit 1
        }
    } else if (exp == 31) {
        // Infinity or NaN
        if (mantissa == 0) {
            // Infinity
            converter.i = sign | 0x7F800000;
            return converter.f;
        } else {
            // NaN
            converter.i = sign | 0x7FC00000;
            return converter.f;
        }
    } else {
        // Normalized number
        exp += 127 - 15; // Adjust bias
        mantissa <<= 13; // Left shift 13 bits
    }
    
    converter.i = sign | (exp << 23) | mantissa;
    return converter.f;
}


// Precise comparison function using fp16 to fp32 conversion
bool Float16Equal(uint16_t a, uint16_t b, float atol = ATOL, float rtol = RTOL)
{
    // Completely equal case
    if (a == b) {
        return true;
    }
    
    // Convert to float for comparison
    float fa = HalfToFloat(a);
    float fb = HalfToFloat(b);
    
    // Handle NaN cases
    if (std::isnan(fa) && std::isnan(fb)) {
        return true;
    }
    if (std::isnan(fa) || std::isnan(fb)) {
        return false;
    }
    
    // Handle infinity cases
    if (std::isinf(fa) && std::isinf(fb)) {
        return (fa > 0) == (fb > 0);
    }
    if (std::isinf(fa) || std::isinf(fb)) {
        return false;
    }
    
    // Use absolute and relative tolerance for comparison
    float diff = std::abs(fa - fb);
    float max_val = std::max(std::abs(fa), std::abs(fb));
    
    return diff <= atol || diff <= rtol * max_val;
}

/**
 * @brief Compare host data of two tensors (float16 type)
 * @param result Result tensor
 * @param expect Expected tensor
 * @param atol Absolute tolerance
 * @param rtol Relative tolerance
 * @return Status Comparison result, 0 means equal, -1 means not equal
 */
atb::Status CompareTensorHostData(const atb::Tensor &result, const atb::Tensor &expect, 
                                  float atol = ATOL, float rtol = RTOL)
{
    if (result.hostData == nullptr || expect.hostData == nullptr) {
        return -1; // ERROR_INVALID_TENSOR_ADDR
    }
    
    if (result.desc.dtype != ACL_FLOAT16 || expect.desc.dtype != ACL_FLOAT16) {
        return -1; // ERROR_INVALID_TENSOR_DTYPE
    }
    
    // Calculate number of elements
    uint64_t resultElements = 1;
    uint64_t expectElements = 1;
    for (uint32_t i = 0; i < result.desc.shape.dimNum; i++) {
        resultElements *= result.desc.shape.dims[i];
    }
    for (uint32_t i = 0; i < expect.desc.shape.dimNum; i++) {
        expectElements *= expect.desc.shape.dims[i];
    }
    
    if (resultElements != expectElements) {
        return -1; // ERROR_INVALID_TENSOR_SIZE
    }
    
    uint16_t *resultData = static_cast<uint16_t*>(result.hostData);
    uint16_t *expectData = static_cast<uint16_t*>(expect.hostData);
    
    for (uint64_t i = 0; i < resultElements; i++) {
        // if (i < 10) {
        //     std::cout << "expectData[" << i << "]: " << expectData[i] << ", resultData[" << i << "]: " << resultData[i] << std::endl;
        // }
        if (!Float16Equal(expectData[i], resultData[i], atol, rtol)) {
            // Output mismatched position and values
            float expectVal = static_cast<float>(expectData[i]);
            float resultVal = static_cast<float>(resultData[i]);
            std::cout << "Data mismatch at position " << i <<"/"<< resultElements
                      << ", expect: " << expectVal 
                      << ", result: " << resultVal << std::endl;
            return -1; // Not equal
        }
    }
    
    return 0; // Equal
}
