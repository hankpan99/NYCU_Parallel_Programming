#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth * sizeof(float);
    int imageSize = imageHeight * imageWidth * sizeof(float);

    // create command queue
    cl_command_queue myqueue;
    myqueue = clCreateCommandQueue(*context, *device, 0, &status);

    // create buffers on device
    cl_mem device_in_img = clCreateBuffer(*context, CL_MEM_READ_ONLY, imageSize, NULL, &status);
    cl_mem device_out_img = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, imageSize, NULL, &status);
    cl_mem device_filter = clCreateBuffer(*context, CL_MEM_READ_ONLY, filterSize, NULL, &status);

    // transfer data from host to device
    status = clEnqueueWriteBuffer(myqueue, device_in_img, CL_TRUE, 0, imageSize, (void*) inputImage, 0, NULL, NULL);
    CHECK(status, "clEnqueueWriteBuffer");
    status = clEnqueueWriteBuffer(myqueue, device_filter, CL_TRUE, 0, filterSize, (void*) filter, 0, NULL, NULL);
    CHECK(status, "clEnqueueWriteBuffer");

    // create kernel function
    cl_kernel mykernel = clCreateKernel(*program, "convolution", &status);
    CHECK(status, "clCreateKernel");

    // set arguments
    clSetKernelArg(mykernel, 0, sizeof(cl_int), (void*) &filterWidth);
    clSetKernelArg(mykernel, 1, sizeof(cl_mem), (void*) &device_filter);
    clSetKernelArg(mykernel, 2, sizeof(cl_int), (void*) &imageHeight);
    clSetKernelArg(mykernel, 3, sizeof(cl_int), (void*) &imageWidth);
    clSetKernelArg(mykernel, 4, sizeof(cl_mem), (void*) &device_in_img);
    clSetKernelArg(mykernel, 5, sizeof(cl_mem), (void*) &device_out_img);

    // set local and global workgroup sizes
    size_t localws[2] = {20, 20};
    size_t globalws[2] = {imageWidth, imageHeight};

    // execute kernel function
    status = clEnqueueNDRangeKernel(myqueue, mykernel, 2, 0, globalws, localws, 0, NULL, NULL);
    CHECK(status, "clEnqueueNDRangeKernel");

    // copy reuslts from device back to host
    status = clEnqueueReadBuffer(myqueue, device_out_img, CL_TRUE, 0, imageSize, (void*) outputImage, NULL, NULL, NULL);
    CHECK(status, "clEnqueueNDRangeKernel");

    // release
    status = clReleaseCommandQueue(myqueue);
    status = clReleaseMemObject(device_in_img);
    status = clReleaseMemObject(device_out_img);
    status = clReleaseMemObject(device_filter);
    status = clReleaseKernel(mykernel);
}