#ifndef KERNEL_H
#define KERNEL_H

// Wrappers
void sumParts(float *out, float *in, int len);

void dotProductLauncher(float *resultGPU, float *resultGPUmem, float *resultBAD, float *time1, float *time2, const float *a, const float *b, int size);

#endif
