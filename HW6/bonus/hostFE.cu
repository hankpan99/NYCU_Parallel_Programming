#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
extern "C"{
#include "hostFE.h"
}
#include "helper.h"

__constant__ float const_filter[500];

__global__ void convolution(int filterWidth, int imageHeight, int imageWidth,
                            float *inputImage, float *outputImage)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    int halffilterSize = filterWidth / 2;
    float sum = 0;

    for(int k = -halffilterSize; k <= halffilterSize; k++){
        for(int l = -halffilterSize; l <= halffilterSize; l++){
            if((iy + k >= 0) && (iy + k < imageHeight) && (ix + l >= 0) && (ix + l < imageWidth)){
                sum += (inputImage[(iy + k) * imageWidth + ix + l] * const_filter[(k + halffilterSize) * filterWidth + l + halffilterSize]);
            }
        }
    }

    outputImage[iy * imageWidth + ix] = sum;
}

extern "C"
void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    int filterSize = filterWidth * filterWidth * sizeof(float);
    int imageSize = imageHeight * imageWidth * sizeof(float);

    // allocate memory
    float *device_in_img, *device_out_img;
    cudaMalloc(&device_in_img, imageSize);
    cudaMalloc(&device_out_img, imageSize);

    // transfer filter to device constant memory (declare with __constant__ float)
    cudaMemcpyToSymbol(const_filter, filter, filterSize, 0, cudaMemcpyHostToDevice);

    // transfer image to deivce global memory
    cudaMemcpy(device_in_img, inputImage, imageSize, cudaMemcpyHostToDevice);

    // launch kernel function
    dim3 threads_per_block(20, 20);
    dim3 num_blocks(imageWidth / threads_per_block.x, imageHeight / threads_per_block.y);
    convolution<<<num_blocks, threads_per_block>>>(filterWidth, imageHeight, imageWidth,
                                                   device_in_img, device_out_img);
    
    // wait for kernel function finish
    cudaDeviceSynchronize();

    // output answers
    cudaMemcpy(outputImage, device_out_img, imageSize, cudaMemcpyDeviceToHost);
    
    // free memory
    cudaFree(device_in_img);
    cudaFree(device_out_img);
}