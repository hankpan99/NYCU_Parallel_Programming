#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mandelKernel(int *device_data, float lowerX, float lowerY, float stepX, float stepY, size_t pitch, int maxIterations){
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    
    // process index
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    // initialize mandel variables
    float c_re = lowerX + thisX * stepX;
    float c_im = lowerY + thisY * stepY;
    float z_re = c_re, z_im = c_im;

    // pointer points to the pixel should be processed in this thread
    int* ptr = (int*) ((char*) device_data + thisY * pitch) + thisX;

    // by theorem in mandel, if |c| <= 0.25 then c belongs to M
    if(z_re * z_re + z_im * z_im <= 0.25f){
        *ptr = maxIterations;
        return;
    }
    
    // mandel iteration
    int intensity;
    for(intensity = 0; intensity < maxIterations; intensity++){
        if(z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }
    
    *ptr = intensity;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE(float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations){
    // compute steps
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    // allocate memory
    int N = resX * resY;
    int *host_data;
    cudaHostAlloc((void**) &host_data, N * sizeof(int), cudaHostAllocMapped);

    int *device_data;
    size_t pitch;
    cudaMallocPitch(&device_data, &pitch, resX * sizeof(int), resY);

    // launch kernel function
    dim3 threads_per_block(20, 20);
    dim3 num_blocks(resX / threads_per_block.x, resY / threads_per_block.y);
    mandelKernel<<<num_blocks, threads_per_block>>>(device_data, lowerX, lowerY, stepX, stepY, pitch, maxIterations);
    
    // output answers
    cudaMemcpy2D(host_data, resX * sizeof(int), device_data, pitch, resX * sizeof(int), resY, cudaMemcpyDeviceToHost);
    memcpy(img, host_data, N * sizeof(int));
    
    // free memory
    cudaFree(device_data);
    cudaFreeHost(host_data);
}