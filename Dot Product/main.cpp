#include "kernel.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#define N 2048*1000


//Main Function

int main(int argc, char *argv[])
{
  // Set values in u2 and v2
  float *u2 = (float*)calloc(N, sizeof(float));
  float *v2 = (float*)calloc(N, sizeof(float));

  for (int i = 0; i < N; i++)
  {
    u2[i] = 1;
    v2[i] = 1;
  }

  float CPUsum = 0;
  float GPUsum = 0;
  float GPUsum_shared = 0;
  float GPUsumBAD = 0;

  float CPU_time = 0;
  float GPU_time = 0;//
  float GPUshared_time = 0;  

  // Run CPU sum

  clock_t CPUBegin = clock();

  float *u2multv2 = (float*)calloc(N, sizeof(float));
  for (int i = 0; i < N; i++)
  {
    u2multv2[i] = u2[i] * v2[i];
    CPUsum += u2multv2[i];
  }

  clock_t CPUEnd = clock();


  CPU_time = 1000*((float)(CPUEnd - CPUBegin))/CLOCKS_PER_SEC;
  printf("CPU Time in ms: %f\n", CPU_time);

  // Run GPU sum, GPU shared memory sum, and GPU bad sum
  dotProductLauncher(&GPUsum, &GPUsum_shared, &GPUsumBAD, &GPU_time, &GPUshared_time, u2, v2, N);
  printf("CPU Sum is: %f\n", CPUsum);
  printf("GPU Sum is: %f\n", GPUsum);
  printf("GPU_Shared is: %f\n", GPUsum_shared);
  printf("Bad GPU Sum is: %f\n", GPUsumBAD);

  // Free memory
  free(u2);
  free(v2);
  free(u2multv2);
  return 0;
}
