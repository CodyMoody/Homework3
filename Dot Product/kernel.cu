#include "kernel.h"
#include <stdio.h>
#define TPB 128

// Component Multiplication
__global__ void componentMultKernel(float *d_out, float *d_a, float *d_b)
{
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  d_out[i] = d_a[i] * d_b[i];
}

// Bad Sum
__global__ void sumKernelBad(float *accum, float *d_in, int size)
{
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= size) return;
  *accum += d_in[i];
}

// Atomic Add Sum
__global__ void sumKernel(float * accum, const float *a, int size)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= size) return;
  atomicAdd(accum, a[i]); // *accum += a[i]
}


// Atomic Add Shared Sum -- Inspired by book example
__global__ void sumKernelShared(float *d_res, const float *d_a, const float *d_b, int n)
{
  const int idx = threadIdx.x*blockDim.x + blockIdx.x;
  if (idx >= n) return;
  const int s_idx = threadIdx.x;

  __shared__ float s_prod[TPB];
  s_prod[s_idx] = d_a[idx] * d_b[idx];
  __syncthreads();
  
  if (s_idx == 0) {
    float blockSum = 0;
    for (int j = 0; j < blockDim.x; ++j) {
      blockSum += s_prod[j];
    }
    //printf("Block_%i, blockSum = %f\n", blockIdx.x, blockSum);
    //*d_res += blockSum;
    atomicAdd(d_res,blockSum);
  }
}

// Problem One Full Encompassing Kernel Launcher
void dotProductLauncher(float *resultGPU, float *resultGPUmem, float *resultBAD, float *time1, float *time2, const float *a, const float *b, int size)
{
  float *d_a = 0;
  float *d_b = 0;

  float *d_mult = 0;
  float *d_accum1 = 0;
  float *d_accum2 = 0;
  float *d_accum3 = 0;
  cudaMalloc(&d_accum1, sizeof(float));
  cudaMalloc(&d_accum2, sizeof(float));
  cudaMalloc(&d_accum3, sizeof(float));
  cudaMalloc(&d_mult, size*sizeof(float));
  cudaMemset(d_accum1, 0, sizeof(float));
  cudaMemset(d_accum2, 0, sizeof(float));
  cudaMemset(d_accum3, 0, sizeof(float));

  cudaMalloc(&d_a, size * sizeof(float));
  cudaMalloc(&d_b, size * sizeof(float));
  cudaMemcpy(d_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size * sizeof(float), cudaMemcpyHostToDevice);

  // For timing the kernels
  cudaEvent_t startKernelMult, stopKernelMult;
  cudaEvent_t startKernelGPU, stopKernelGPU;
  cudaEvent_t startKernelGPUshared, stopKernelGPUshared;
  cudaEventCreate(&startKernelMult);
  cudaEventCreate(&stopKernelMult);
  cudaEventCreate(&startKernelGPU);
  cudaEventCreate(&stopKernelGPU);
  cudaEventCreate(&startKernelGPUshared);
  cudaEventCreate(&stopKernelGPUshared);

  cudaEventRecord(startKernelMult);
  componentMultKernel<<<(size + TPB - 1)/TPB, TPB>>>(d_mult, d_a, d_b);
  cudaEventRecord(stopKernelMult);

  // Atomic Add (GOOD) Computation
  cudaEventRecord(startKernelGPU);
  sumKernel<<<(size + TPB - 1)/TPB, TPB>>>(d_accum1, d_mult, size);
  cudaEventRecord(stopKernelGPU);

  cudaMemcpy(resultGPU, d_accum1, sizeof(float), cudaMemcpyDeviceToHost);

  // Atomic Add Computation with Shared Memory
  cudaEventRecord(startKernelGPUshared);
  sumKernelShared<<<(size + TPB - 1)/TPB, TPB>>>(d_accum2, d_a, d_b, size);
  cudaEventRecord(stopKernelGPUshared);

  cudaMemcpy(resultGPUmem, d_accum2, sizeof(float), cudaMemcpyDeviceToHost);
  
  // BAD Computation
  sumKernelBad<<<(size + TPB - 1)/TPB, TPB>>>(d_accum3, d_mult, size);

  cudaMemcpy(resultBAD, d_accum3, sizeof(float), cudaMemcpyDeviceToHost);

  // Synchronize Cuda Events
  cudaEventSynchronize(stopKernelMult);
  cudaEventSynchronize(stopKernelGPU);
  cudaEventSynchronize(stopKernelGPUshared);

  float multTimeInMs = 0;
  float gpuTimeInMs = 0;
  float gpuSharedTimeInMs = 0;

  cudaEventElapsedTime(&multTimeInMs, startKernelMult, stopKernelMult);
  cudaEventElapsedTime(&gpuTimeInMs, startKernelGPU, stopKernelGPU);
  cudaEventElapsedTime(&gpuSharedTimeInMs, startKernelGPUshared, stopKernelGPUshared);

  gpuTimeInMs += multTimeInMs;

  printf("GPU Time in ms: %f\n", gpuTimeInMs);
  printf("GPU Shared Memory Time in ms: %f\n", gpuSharedTimeInMs);

  cudaFree(d_accum1);
  cudaFree(d_accum2);
  cudaFree(d_accum3);
  cudaFree(d_mult);
  cudaFree(d_a);
  cudaFree(d_b);
}
