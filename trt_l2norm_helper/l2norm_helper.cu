#include "l2norm_helper.h"


__global__ void sqrtKernel(
    const int n,
    const float* x,
    float* y)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x)
    {
        y[i] = sqrtf(x[i]);
    }
}

__global__ void rsqrtKernel(
    const int n,
    const float* x,
    float* y)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x)
    {
        y[i] = rsqrtf(x[i]);
    }
}

__global__ void maxKernel(
    const int n,
    const float eps,
    const float* x,
    float* y)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x)
    {
        y[i] = fmaxf(x[i], eps);
    }
}

bool executeInference(
    cudaStream_t stream,
    const int op_type,
    const float eps,
    const int batch_size,
    const int C,
    const int H,
    const int W,
    const void* inputData,
    void* outputData)
{
    const int length = C * H * W;
    float* input = (float*) const_cast<void*>(inputData);
    float* output = (float*) outputData;
    for (int n = 0; n < batch_size; ++n)
    {
        switch(op_type)
        {
          case operation_t::OP_TYPE_MAX:
            maxKernel<<<(length + 511) / 512, 512, 0, stream>>>(length, eps, input, output);
            break;
          case operation_t::OP_TYPE_RSQRT:
            rsqrtKernel<<<(length + 511) / 512, 512, 0, stream>>>(length, input, output);
            break;
          case operation_t::OP_TYPE_SQRT:
            sqrtKernel<<<(length + 511) / 512, 512, 0, stream>>>(length, input, output);
            break;
          default:
            return 1;
        }
        // Move cursors
        input += length;
        output += length;
    }
    return 0;
}
